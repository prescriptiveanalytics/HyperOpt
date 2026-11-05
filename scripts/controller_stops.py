"""Controller stop behaviour of agent jobs.

Per job: the trials, the median run length, the runs the controller stopped, the runs stopped at their first decision
(split by the run's position against earlier runs at the same evaluations: rank 1, better than their median, median or
worse; read from the AtSameEvaluations field of the first controller request, so '-' for prompts without it), stops right
after a checkpoint with at least 10% improvement ("stops while improving"), and the share of decisions that agree with
the 1% stagnation rule (the rule never stops at a run's first decision, as it needs a previous checkpoint).

With --by-population: per model and variant, runs and stops by population size (100-399, 400-699, 700-1000), since
checkpoints every E/10 evaluations give large populations fewer generations before each decision (3 at population 1000
and E = 30,000).

With --hazard: per model and variant, the stop rate at the k-th decision of a run (stops / runs that reached decision k),
for k while at least --min-runs runs reached it, the remaining decisions pooled into the last value. The line ends with
the --stop-hazard argument for the matched random-stopping control (AgentParameter.Controller "random").

Reads every *.protocol.jsonl below the given directories that ends with job_completed (attempts that were interrupted or
failed are skipped, so a rerun is never mixed with its original). Fallback decisions are not decisions of the controller
and are left out. Standard library only. Used for the controller prompt pilot (Supplement S4):

    python scripts/controller_stops.py results/cp results/pilots/2026-11-04-nofinish
    python scripts/controller_stops.py --hazard results/repilot
    python scripts/controller_stops.py --by-population results/repilot
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics

ORDER = ['final', 'standard', 'p1', 'p2', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'rule', 'random']
IMPROVING = 0.10
RULE = 0.01


def relative_improvement(progress):
    """Reduction of the best training NMSE since the previous checkpoint, relative to its previous value."""
    now, recent = progress.get('BestTrainingNmse'), progress.get('RecentImprovement')
    if now is None or recent is None or isinstance(now, str) or isinstance(recent, str) or now + recent <= 0:
        return None
    return recent / (now + recent)


def job(path):
    name = os.path.basename(path)
    model = re.search(r'model=qwen3\.5(?:%3A|:)(\w+)-q4', name)
    if model is None:   # not an agent job (CPU baselines)
        return None
    model = model.group(1)
    # The job_started record names the prompt since 4 November; before, the job name carried it, and jobs without a
    # suffix ran the then-default "standard" prompt.
    prompt = re.search(r'cprompt=([a-z0-9]+)', name)
    prompt = prompt.group(1) if prompt else 'standard'
    variant = ''
    used, best, stopped = [], [], 0
    populations = []   # (population, stopped by the controller, stopped at its first decision) per trial
    first = collections.Counter()
    decisions = collections.defaultdict(list)   # trial -> actions of the controller's own decisions, in order
    improving_stops = agree = compared = 0
    seen, last_input, seconds = set(), None, None
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            record = json.loads(line)
            kind, data = record['Type'], record['Data']
            if kind == 'job_started':
                agent = data.get('Agent') or {}
                prompt = agent.get('ControllerPrompt') or prompt
                if agent.get('Controller') not in (None, 'llm'):
                    prompt = agent['Controller']
                variant = data.get('Variant') or ''
                if variant.startswith('feedback=final'):
                    prompt = 'final'
            elif kind == 'controller_request':
                last_input = data['Input']
            elif kind == 'agent_trial':
                e = data['Experiment']
                used.append(e['EvaluationsUsed'])
                if not isinstance(e['BestFitness'], str):
                    best.append(e['BestFitness'])
                stopped += e['TerminationReason'] == 'controller_stop'
                actions = decisions.get(data['Trial'], [])
                populations.append((e['HyperParameters']['PopulationSize'], e['TerminationReason'] == 'controller_stop',
                                    bool(actions) and actions[0] == 'Stop'))
            elif kind == 'checkpoint':
                observation = data['Observation']
                if observation.get('UsedFallback'):
                    continue
                trial, action = data['Trial'], observation['Decision']['Action']
                decisions[trial].append(action)
                gain = relative_improvement(observation['Progress'])
                if action == 'Stop' and gain is not None and gain >= IMPROVING:
                    improving_stops += 1
                rule_stops = len(decisions[trial]) > 1 and gain is not None and gain < RULE
                agree += (action == 'Stop') == rule_stops
                compared += 1
                if trial in seen:
                    continue
                seen.add(trial)
                match = re.search(r'"AtSameEvaluations":\{([^}]*)\}', last_input or '')
                if not match:
                    position = 'none'
                else:
                    comparison = json.loads('{' + match.group(1) + '}')
                    if comparison['ThisRunRank'] == 1:
                        position = 'rank1'
                    elif observation['Progress']['BestTrainingNmse'] < comparison['Median']:
                        position = 'better'
                    else:
                        position = 'worse'
                first[(position, action)] += 1
            elif kind == 'job_completed':
                seconds = data['Timing']['TotalJobMs'] / 1000
    if seconds is None:
        return None
    return {'model': model, 'prompt': prompt, 'variant': variant, 'used': used, 'best': best, 'stopped': stopped,
            'first': first, 'seconds': seconds, 'decisions': decisions, 'improving_stops': improving_stops,
            'agree': agree, 'compared': compared, 'populations': populations}


def share(first, position):
    stops = first[(position, 'Stop')]
    n = stops + first[(position, 'Continue')]
    return f'{stops}/{n}' if n else '-'


def result_protocols(directory):
    """Protocol files named by the results in a directory (ProtocolFile); None where no result names one (older pilots)."""
    named = set()
    for path in glob.glob(os.path.join(directory, '**', '*.result.json'), recursive=True):
        with open(path, encoding='utf-8-sig') as handle:
            protocol = json.load(handle).get('ProtocolFile')
        if protocol:
            named.add(os.path.join(os.path.dirname(path), protocol))
    return named or None


def completed_jobs(directories):
    for directory in directories:
        # Where results name their protocol, only those count: a rerun after a deleted result leaves the original's
        # completed protocol behind, which must not be mixed with the rerun.
        named = result_protocols(directory)
        for path in sorted(glob.glob(os.path.join(directory, '**', '*.protocol.jsonl'), recursive=True)):
            if named is not None and os.path.normcase(os.path.abspath(path)) not in {os.path.normcase(os.path.abspath(n)) for n in named}:
                continue
            result = job(path)
            if result is not None and result['used']:
                yield result


def table(directories):
    rows = []
    for j in completed_jobs(directories):
        first = j['first']
        first_stops = sum(v for (_, action), v in first.items() if action == 'Stop')
        total_stops = sum(a == 'Stop' for actions in j['decisions'].values() for a in actions)
        rows.append((j['model'], j['prompt'], len(j['used']), int(statistics.median(j['used'])), f"{j['stopped']}/{len(j['used'])}",
                     f'{first_stops}/{sum(first.values())}', share(first, 'rank1'), share(first, 'better'), share(first, 'worse'),
                     f"{j['improving_stops']}/{total_stops}", f"{j['agree']}/{j['compared']}",
                     f"{min(j['best']):.4f}" if j['best'] else '-', f"{j['seconds']:.0f}"))
    print('\t'.join(['model', 'prompt', 'trials', 'median run', 'stopped', 'stopped at first decision',
                     'first: rank 1', 'first: better than median', 'first: median or worse',
                     'stops while improving (>=10%)', 'decisions agreeing with 1% rule', 'best valNMSE', 'seconds']))
    for row in sorted(rows, key=lambda r: (r[0], ORDER.index(r[1]) if r[1] in ORDER else len(ORDER))):
        print('\t'.join(str(v) for v in row))


def hazards(directories, min_runs):
    """Stop rate at the k-th decision per (model, variant), pooled over jobs; the tail is pooled into the last value."""
    groups = collections.defaultdict(list)
    for j in completed_jobs(directories):
        groups[(j['model'], j['variant'])].extend(j['decisions'].values())
    print('\t'.join(['model', 'variant', 'runs', 'stop rate per decision (stops/reached)', '--stop-hazard']))
    for (model, variant), runs in sorted(groups.items()):
        rates, cells, k = [], [], 0
        while True:
            reached = [actions for actions in runs if len(actions) > k]
            if not reached:
                break
            if len(reached) < min_runs and rates:
                # Pool the remaining decisions into the last value.
                tail = [a for actions in reached for a in actions[k:]]
                stops = sum(a == 'Stop' for a in tail)
                cells.append(f'{k + 1}+: {stops}/{len(tail)}')
                rates.append(stops / len(tail))
                break
            stops = sum(actions[k] == 'Stop' for actions in reached)
            cells.append(f'{k + 1}: {stops}/{len(reached)}')
            rates.append(stops / len(reached))
            k += 1
        print('\t'.join([model, variant, str(len(runs)), ', '.join(cells), ','.join(f'{r:.3f}'.rstrip('0').rstrip('.') or '0' for r in rates)]))


BINS = [(100, 399), (400, 699), (700, 1000)]


def by_population(directories):
    groups = collections.defaultdict(list)
    for j in completed_jobs(directories):
        groups[(j['model'], j['variant'])].extend(j['populations'])
    print('	'.join(['model', 'variant'] + [f'population {lo}-{hi}: stopped / first-decision stops / runs' for lo, hi in BINS]))
    for (model, variant), trials in sorted(groups.items()):
        cells = []
        for lo, hi in BINS:
            inside = [t for t in trials if lo <= t[0] <= hi]
            cells.append(f'{sum(t[1] for t in inside)} / {sum(t[2] for t in inside)} / {len(inside)}')
        print('	'.join([model, variant] + cells))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--hazard', action='store_true', help='stop rate per decision index, per model and variant')
    parser.add_argument('--by-population', action='store_true', help='runs and stops by population size, per model and variant')
    parser.add_argument('--min-runs', type=int, default=10, help='decisions reached by fewer runs are pooled (default 10)')
    args = parser.parse_args()
    if args.hazard:
        hazards(args.directories, args.min_runs)
    elif args.by_population:
        by_population(args.directories)
    else:
        table(args.directories)
