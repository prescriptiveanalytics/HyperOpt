"""Primary outcomes of a campaign from its *.result.json files (standard library only).

Every tuning run records snapshots of its best expression every E/10 evaluations and at its end. Selection takes the
snapshot with the lowest validation NMSE, values below SOLVED counting as equal, then the shortest expression, then the
earliest (trial order, then evaluations) - the rule of HyperParameterOptimizationAlgorithm.SelectExpression. Replayed
over cumulative evaluations, it gives the expression a method would have selected at any point of its budget.

Outputs (CSV, in --out):
  jobs.csv      one row per job: final selected test NMSE (floored), solved, evaluations to each target, budget used
  summary.csv   per method and task: median floored log10 test NMSE, solve rate, ERT per target
  ecdf.csv      per method: share of (task, target, repetition) triples reached within each evaluation count
  anytime.csv   per method and task: median floored log10 test NMSE of the selection at each evaluation count
  contrasts.csv with --compare: task-weighted paired contrasts, Holm-corrected per family (see contrasts())
summary.csv also holds 95% bootstrap intervals of the ERT (repetitions resampled within each task, fixed seed).
Methods are grouped by algorithm, model, variant and target noise. Noiseless jobs use the targets 1e-2..1e-10 and
"solved" means test NMSE < 1e-10; noisy jobs use 1e-2..1e-6 (fits cannot reach 1e-10 on noisy data).

Failures and retries follow the declared policy (scripts/attempts.py): every job of expected-jobs.txt must resolve to a
result or a counted failure, or the analysis stops (--allow-incomplete for partial looks at a running study, which are
not for inference). All results must share one code commit, fitting setting and model digest per model (--allow-mixed for
development data only).

Scoring: log10 test NMSE floored at 1e-10 (solved) and capped at LOG_CAP (NMSE 100); a failed job, a non-finite NMSE and
"no expression selected yet" all score LOG_CAP, the worst value, so a failure never ranks above a valid fit.

Usage:  python scripts/analyse_outcomes.py <result directory> [more directories] --out <output directory>
            [--compare "RQ1:A~B" --compare "RQ1-further:A~C,B~D//A~C"]
        A~B is the paired difference A - B; A~B//C~D the interaction (A - B) - (C - D) (method names contain '|'). A, B, C, D are substrings that
        must each match exactly one method name (see summary.csv).
        python scripts/analyse_outcomes.py --self-test
"""
import argparse, csv, glob, json, math, os, random, statistics, sys

SOLVED = 1e-10
NOISELESS_TARGETS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
NOISY_TARGETS = [1e-2, 1e-4, 1e-6]


LOG_CAP = 2.0   # log10 NMSE 100: valid errors are capped here, and failures and missing values score it


def floor_log(value):
    """log10 of NMSE floored at SOLVED and capped at LOG_CAP; None and non-finite values (failure, no expression yet) score
    LOG_CAP, the worst value, so they never rank above a valid fit (a capped NMSE above 100 ties with them)."""
    if value is None or not math.isfinite(value):
        return LOG_CAP
    return min(LOG_CAP, math.log10(max(value, SOLVED)))


def method_key(result):
    return (result['AlgorithmName'], result.get('ModelName') or '', result.get('Variant') or '', result.get('TargetNoise') or 0.0,
            result.get('SampleRows') or 0)


def method_name(key):
    algorithm, model, variant, noise, rows = key
    return '|'.join(part for part in (algorithm, model, variant, f'noise={noise:g}' if noise else '', f'rows={rows}' if rows else '') if part)


def snapshots_in_order(result):
    """(cumulative evaluations, trial index, snapshot) for every snapshot, trials in execution order."""
    entries, spent = [], 0
    for index, trial in enumerate(result['ConfigurationResults']):
        for snapshot in trial.get('Snapshots') or []:
            entries.append((spent + snapshot['Evaluations'], index, snapshot))
        spent += trial['EvaluationsUsed']
    return entries, spent


def selection_key(entry):
    _, index, snapshot = entry
    validation = snapshot['ValidationNmse']
    return (max(validation, SOLVED) if math.isfinite(validation) else math.inf, snapshot['Length'], index, snapshot['Evaluations'])


def selection_path(result):
    """The selected snapshot after each new snapshot: [(cumulative evaluations, test NMSE, noisy test NMSE)]."""
    entries, spent = snapshots_in_order(result)
    path, best = [], None
    for entry in sorted(entries, key=lambda e: e[0]):
        if best is None or selection_key(entry) < selection_key(best):
            best = entry
        path.append((entry[0], best[2]['TestNmse'], best[2].get('NoisyTestNmse')))
    return path, spent


def evaluations_to(path, target):
    """First cumulative evaluation count at which the selected expression's test NMSE is below target; None if never."""
    for evaluations, test, _ in path:
        if test is not None and math.isfinite(test) and test < target:
            return evaluations
    return None


def analyse_job(result):
    path, spent = selection_path(result)
    if not path:
        raise ValueError(f"{result['AlgorithmName']} {result['InstanceName']} rep {result['Repetition']}: no snapshots (result predates snapshot logging)")
    noisy = bool(result.get('TargetNoise'))
    targets = NOISY_TARGETS if noisy else NOISELESS_TARGETS
    final_test, final_noisy = path[-1][1], path[-1][2]
    recorded = result.get('ExpressionSelection')
    if recorded is not None and recorded['TestNmse'] != final_test:
        raise ValueError(f"{result['AlgorithmName']} {result['InstanceName']} rep {result['Repetition']}: replayed selection "
                         f"(test NMSE {final_test}) differs from the runner's ExpressionSelection ({recorded['TestNmse']})")
    budget = result['NumConfigurations'] * result['EvaluationsPerConfiguration']
    # Overfitting check of the noisy variant: final validation minus training NMSE of each trial's best expression.
    gaps = [trial['BestFitness'] - trial['TrainingBestFitness'] for trial in result['ConfigurationResults']
            if trial.get('TrainingBestFitness') is not None and math.isfinite(trial['BestFitness']) and math.isfinite(trial['TrainingBestFitness'])]
    return {
        'method': method_name(method_key(result)), 'task': result['InstanceName'], 'repetition': result['Repetition'],
        'noisy': noisy, 'path': path, 'spent': spent, 'budget': budget, 'trials': len(result['ConfigurationResults']),
        'final_test': final_test, 'final_noisy_test': final_noisy, 'final_log10': floor_log(final_test),
        'solved': (final_test is not None and math.isfinite(final_test) and final_test < SOLVED) and not noisy,
        'targets': targets, 'hits': {target: evaluations_to(path, target) for target in targets},
        'median_validation_minus_training': statistics.median(gaps) if gaps else '',
        'share_validation_above_training': sum(g > 0 for g in gaps) / len(gaps) if gaps else '',
        'sample_rows': result.get('SampleRows') or 0,
        'failed': False, **search_columns(result.get('Search')), **reliability_columns(result), **cost_columns(result),
    }


def reliability_columns(result):
    """Configuration reliability: the nominee's fresh-seed reruns (K seeds at E and at 4E), median floored log10 test NMSE
    of the final training-best expression per length (assessment runs record no snapshots)."""
    runs = result.get('ConfigurationAssessmentResults') or []
    columns = {}
    for label, length in (('E', result['EvaluationsPerConfiguration']), ('4E', 4 * result['EvaluationsPerConfiguration'])):
        values = [floor_log(r.get('TestFitness')) for r in runs if r.get('EvaluationsLimit') == length]
        columns[f'reliability_{label}_median_log10'] = statistics.median(values) if values else ''
        columns[f'reliability_{label}_runs'] = len(values)
    return columns


def cost_columns(result):
    """Model requests per role from the job's own protocol (ProtocolFile): calls, prompt and completion tokens, and HTTP
    time. Decisions of non-LLM controllers make no request; HTTP time includes waiting behind concurrent jobs."""
    columns = {f'{role}_{field}': '' for role in ('configurator', 'controller')
               for field in ('calls', 'prompt_tokens', 'completion_tokens', 'http_seconds')}
    path = result.get('_path')
    if not path or not result.get('ProtocolFile'):
        return columns
    protocol = os.path.join(os.path.dirname(path), result['ProtocolFile'])
    if not os.path.exists(protocol):
        return columns
    totals = {role: [0, 0, 0, 0.0] for role in ('configurator', 'controller')}
    with open(protocol, encoding='utf-8-sig') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('Type') != 'llm_call' or record['Data'].get('Role') not in totals:
                continue
            data, total = record['Data'], totals[record['Data']['Role']]
            total[0] += 1
            total[1] += data.get('InputTokens') or 0
            total[2] += data.get('OutputTokens') or 0
            total[3] += (data.get('DurationMs') or 0) / 1000
    for role, (calls, prompt, completion, seconds) in totals.items():
        columns.update({f'{role}_calls': calls, f'{role}_prompt_tokens': prompt, f'{role}_completion_tokens': completion,
                        f'{role}_http_seconds': round(seconds, 1)})
    return columns


def search_columns(search):
    """How an agent search ended (result field Search, or the job_failed record): exit, evaluations left unspent, rejected
    replies and proposals or allowances moved into the bounds; empty for the other methods."""
    search = search or {}
    return {'search_exit': search.get('Exit', ''), 'unspent_evaluations': search.get('RemainingEvaluations', ''),
            'invalid_replies': search.get('InvalidReplies', ''), 'clamped_proposals': search.get('ClampedProposals', ''),
            'clamped_allowances': search.get('ClampedAllowances', '')}


def failed_job(started, failure):
    """A job that failed without a result (an agent search with no valid configuration): it counts as unsolved with the
    worst floored test NMSE and no target reached, so paired tests do not silently drop it."""
    key = (started['Algorithm'], started.get('Model') or '', started.get('Variant') or '', started.get('TargetNoise') or 0.0,
           started.get('SampleRows') or 0)
    noisy = bool(started.get('TargetNoise'))
    targets = NOISY_TARGETS if noisy else NOISELESS_TARGETS
    return {'method': method_name(key), 'task': started['Instance'], 'repetition': started['Repetition'], 'noisy': noisy,
            'path': [], 'spent': 0, 'budget': started['T'] * started['E'], 'trials': 0, 'final_test': None,
            'final_noisy_test': None, 'final_log10': floor_log(None), 'solved': False, 'targets': targets,
            'hits': {target: None for target in targets}, 'median_validation_minus_training': '',
            'share_validation_above_training': '', 'sample_rows': started.get('SampleRows') or 0, 'failed': True,
            **search_columns(failure.get('Search')),
            **{f'reliability_{l}_{f}': '' for l in ('E', '4E') for f in ('median_log10', 'runs')},
            **cost_columns({})}


def load_studies(directories, allow_incomplete=False, allow_mixed=False):
    """Results and counted failures of each directory from scripts/attempts.py load_study, the same check as the validator:
    the failure and retry policy, completeness against expected-jobs.txt, results tied to their completed protocols, and
    identity against study-identity.json. Stops on completeness problems unless allow_incomplete and on identity problems
    unless allow_mixed."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import attempts
    results, failed, notes, signatures = [], [], [], {}
    for directory in directories:
        study = attempts.load_study(directory)
        signatures[directory] = attempts.identity_signature(directory, study)
        counts = study['counts']
        notes.append(f"{directory}: {counts['result']} results, {counts['failure']} counted failures, {counts['unresolved']} unresolved")
        if study['completeness'] and not allow_incomplete:
            sys.exit(f"{notes[-1]}; {study['completeness'][0]}. Complete the study (or --allow-incomplete for a look, not for inference).")
        if study['identity'] and not allow_mixed:
            sys.exit(f"{directory}: {study['identity'][0]} (and {len(study['identity']) - 1} more); --allow-mixed for development data only.")
        for result in study['results']:
            results.append(json.loads(json.dumps(result), object_hook=named_floats))
        failed += [failed_job(started, failure) for _, started, failure in study['failures'] if started]
    # Each directory is checked on its own above; directories analysed together must also agree with one another.
    across = attempts.cross_directory_problems(signatures) if len(signatures) > 1 else []
    if across and not allow_mixed:
        sys.exit(f"the directories do not share one identity: {across[0]} (and {len(across) - 1} more); --allow-mixed for development data only.")
    return results, failed, notes


def ert(jobs, target):
    """Expected running time (COCO): evaluations over all runs per success. Unsuccessful runs are charged the full budget B,
    also when they ended early, so budget a method leaves unused counts against it, never for it."""
    successes = [job['hits'][target] for job in jobs if job['hits'].get(target) is not None]
    if not successes:
        return math.inf
    spent = sum(job['hits'][target] if job['hits'].get(target) is not None else job['budget'] for job in jobs)
    return spent / len(successes)


def ert_interval(jobs, target, resamples=1000, seed=1):
    """95% percentile bootstrap interval of the ERT, resampling repetitions with replacement; inf where no resample succeeds."""
    rng = random.Random(seed)
    values = sorted(ert([rng.choice(jobs) for _ in jobs], target) for _ in range(resamples))
    return values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]


def wilcoxon(differences):
    """Two-sided Wilcoxon signed-rank test with Pratt's treatment of zeros: absolute differences are ranked with the zeros
    (tied values get their average rank), and the zeros' ranks are left out of the statistic. Exact distribution of W+
    over all sign assignments of the nonzero differences, also with ties (ranks are multiples of 1/2). Returns
    (number of nonzero differences, W+, p)."""
    values = list(differences)
    n_all = len(values)
    order = sorted(range(n_all), key=lambda i: abs(values[i]))
    ranks = [0.0] * n_all
    i = 0
    while i < n_all:
        j = i
        while j + 1 < n_all and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    nonzero = [(r, x) for r, x in zip(ranks, values) if x != 0]
    if not nonzero:
        return 0, 0.0, 1.0
    doubled = [int(round(2 * r)) for r, _ in nonzero]
    w_plus = sum(r for r, x in nonzero if x > 0)
    counts = {0: 1}                                   # sign assignments per doubled W+ value
    for d in doubled:
        nxt = dict(counts)
        for total, c in counts.items():
            nxt[total + d] = nxt.get(total + d, 0) + c
        counts = nxt
    observed, all_count = int(round(2 * w_plus)), 2 ** len(nonzero)
    low = sum(c for t, c in counts.items() if t <= observed)
    high = sum(c for t, c in counts.items() if t >= observed)
    return len(nonzero), w_plus, min(1.0, 2 * min(low, high) / all_count)


def mcnemar(only_a, only_b):
    """Exact two-sided McNemar test from the discordant pair counts (binomial with p = 1/2)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(p_values):
    """Holm-adjusted p-values in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted, running = [0.0] * len(p_values), 0.0
    for position, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - position) * p_values[i]))
        adjusted[i] = running
    return adjusted


def resolve(fragment, methods):
    """The method with exactly this name, else the only method whose name contains the fragment."""
    if fragment in methods:
        return fragment
    matches = [m for m in methods if fragment in m]
    if len(matches) != 1:
        raise ValueError(f"'{fragment}' must match exactly one method, matched {matches or 'none'}")
    return matches[0]


def contrasts(jobs, families, resamples=2000, seed=7):
    """Task-weighted paired contrasts (the estimand: the mean over the thirteen tasks, each weighted equally, of the mean
    paired difference over the repetitions both conditions share on that task). A~B is A - B in floored, capped log10 test
    NMSE (negative: A better); A~B//C~D the interaction (A - B) - (C - D), on the repetitions all four share per task (for
    model sizes, the common repetitions 1-5). Two statements, labelled separately: (1) the effect estimate, the
    task-weighted mean, with a 95% interval for this fixed suite of tasks: the tasks are held fixed and the paired
    repetitions are resampled within each task, so the interval reflects GP and LLM randomness, not the choice of tasks;
    (2) the exact two-sided Wilcoxon signed-rank test (Pratt) of the per-task differences, which tests whether their
    distribution is symmetric about zero (not that their mean is zero), so the two need not agree. The solve-rate
    difference (noiseless only) is reported descriptively, task-weighted. Holm correction within each family."""
    methods = sorted({job['method'] for job in jobs})
    by_key = {(job['method'], job['task'], job['repetition']): job for job in jobs}
    rng = random.Random(seed)
    rows = []
    for family in families:
        name, _, items = family.partition(':')
        family_rows = []
        for item in items.split(','):
            parts = [part.split('~') for part in item.split('//')]
            terms = [(resolve(a.strip(), methods), resolve(b.strip(), methods)) for a, b in parts]
            used = {m for pair in terms for m in pair}
            per_task, solved_diff, pairs = {}, {}, 0
            for task in sorted({job['task'] for job in jobs}):
                reps = sorted(set.intersection(*({k[2] for k in by_key if k[0] == m and k[1] == task} for m in used)))
                if not reps:
                    continue
                def value(rep, field):
                    a, b = terms[0]
                    v = by_key[(a, task, rep)][field] - by_key[(b, task, rep)][field]
                    if len(terms) == 2:
                        c, d = terms[1]
                        v -= by_key[(c, task, rep)][field] - by_key[(d, task, rep)][field]
                    return float(v)
                per_task[task] = [value(r, 'final_log10') for r in reps]
                if not any(by_key[(m, task, r)]['noisy'] for m in used for r in reps):
                    solved_diff[task] = statistics.mean(value(r, 'solved') for r in reps)
                pairs += len(reps)
            means = [statistics.mean(v) for v in per_task.values()]
            n, w_plus, p = wilcoxon(means)
            boot = []
            tasks = list(per_task)
            for _ in range(resamples if tasks else 0):
                boot.append(statistics.mean(statistics.mean(rng.choice(per_task[t]) for _ in per_task[t]) for t in tasks))
            boot.sort()
            unpaired = sum(1 for k in by_key if k[0] in used and k[1] in per_task) - pairs * len(used)
            family_rows.append({
                'family': name, 'contrast': item.strip(), 'terms': ' // '.join(f'{a} ~ {b}' for a, b in terms),
                'tasks': len(means), 'pairs': pairs, 'unpaired_jobs': unpaired,
                'mean_task_difference': statistics.mean(means) if means else '',
                'ci_low': boot[int(0.025 * len(boot))] if boot else '', 'ci_high': boot[int(0.975 * len(boot)) - 1] if boot else '',
                'median_task_difference': statistics.median(means) if means else '',
                'tasks_a_better': sum(m < 0 for m in means), 'tasks_b_better': sum(m > 0 for m in means),
                'nonzero_tasks': n, 'wilcoxon_w_plus': w_plus, 'p': p,
                'solve_rate_difference': statistics.mean(solved_diff.values()) if solved_diff else ''})
        for row, adjusted in zip(family_rows, holm([row['p'] for row in family_rows])):
            row['p_holm'] = adjusted
        rows.extend(family_rows)
    return rows


def evaluation_grid(budget, points=60):
    """Log-spaced evaluation counts from 1,000 to the budget, including the budget."""
    low = 1000
    return sorted({int(round(low * (budget / low) ** (i / (points - 1)))) for i in range(points)} | {budget})


def selected_at(path, evaluations):
    test = None
    for count, value, _ in path:
        if count > evaluations:
            break
        test = value
    return test


def analyse(results, failed=()):
    jobs = [analyse_job(result) for result in results] + list(failed)
    # Each (method, task, repetition) must occur once; a duplicate would silently replace a job in the paired tests.
    seen = set()
    for job in jobs:
        key = (job['method'], job['task'], job['repetition'])
        if key in seen:
            raise ValueError(f"duplicate job {key}: analyse each campaign directory once and keep variants apart")
        seen.add(key)
    by_method_task = {}
    for job in jobs:
        by_method_task.setdefault((job['method'], job['task']), []).append(job)
    summary = []
    for (method, task), group in sorted(by_method_task.items()):
        targets = group[0]['targets']
        row = {'method': method, 'task': task, 'repetitions': len(group),
               'median_log10_test': statistics.median(job['final_log10'] for job in group),
               'solve_rate': sum(job['solved'] for job in group) / len(group) if not group[0]['noisy'] else '',
               'mean_budget_used': statistics.mean(job['spent'] / job['budget'] for job in group)}
        for target in targets:
            row[f'ert_{target:g}'] = ert(group, target)
            row[f'ert_{target:g}_ci_low'], row[f'ert_{target:g}_ci_high'] = ert_interval(group, target)
        summary.append(row)
    ecdf, anytime = [], []
    by_method = {}
    for job in jobs:
        by_method.setdefault(job['method'], []).append(job)
    for method, group in sorted(by_method.items()):
        grid = evaluation_grid(max(job['budget'] for job in group))
        triples = [(job, target) for job in group for target in job['targets']]
        for evaluations in grid:
            reached = sum(1 for job, target in triples if job['hits'][target] is not None and job['hits'][target] <= evaluations)
            ecdf.append({'method': method, 'evaluations': evaluations, 'share_reached': reached / len(triples)})
    for (method, task), group in sorted(by_method_task.items()):
        for evaluations in evaluation_grid(max(job['budget'] for job in group)):
            anytime.append({'method': method, 'task': task, 'evaluations': evaluations,
                            'median_log10_test': statistics.median(floor_log(selected_at(job['path'], evaluations)) for job in group)})
    return jobs, summary, ecdf, anytime


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def named_floats(entry):
    """The runner writes non-finite numbers as the strings "Infinity", "-Infinity" and "NaN" (AllowNamedFloatingPointLiterals)."""
    return {key: float(value) if value in ('Infinity', '-Infinity', 'NaN') else value for key, value in entry.items()}


def load(directories):
    results = []
    for directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, '*.result.json'))):
            with open(path, encoding='utf-8-sig') as handle:
                result = json.load(handle, object_hook=named_floats)
            result['_path'] = path
            results.append(result)
    return results


def self_test():
    """Hand-checked example: selection, floor, tie-break, evaluations to targets, ERT and ECDF."""
    def snap(evaluations, validation, test, length, final=False):
        return {'Evaluations': evaluations, 'ValidationNmse': validation, 'TestNmse': test, 'Length': length, 'Final': final}
    def job(rep, trials):
        return {'AlgorithmName': 'M', 'InstanceName': 'T', 'Repetition': rep, 'NumConfigurations': 2, 'EvaluationsPerConfiguration': 10000,
                'ConfigurationResults': [{'EvaluationsUsed': used, 'Snapshots': snaps} for used, snaps in trials]}
    # Rep 1: trial 0 (10k) reaches 1e-5 at 6k; trial 1 starts at 10k and hits 1e-20 at 13k (validation below the floor),
    # then 1e-25 with a longer expression at 16k, which the tie-break must not select.
    rep1 = job(1, [(10000, [snap(3000, 9e-3, 9e-3, 9), snap(6000, 1e-5, 1e-5, 9), snap(10000, 1e-5, 1e-5, 9, True)]),
                   (10000, [snap(3000, 1e-20, 1e-21, 11), snap(6000, 1e-25, 1e-26, 15), snap(10000, 1e-25, 1e-26, 15, True)])])
    # Rep 2: never better than 1e-3, and ends after 15k of its 20k budget; failures are charged the full budget.
    rep2 = job(2, [(15000, [snap(3000, 1e-3, 1e-3, 7), snap(15000, 1e-3, 1e-3, 7, True)])])
    jobs, summary, ecdf, anytime = analyse([rep1, rep2])
    first, second = jobs
    assert first['path'][-1][1] == 1e-21, 'the shorter expression below the floor stays selected'
    assert first['hits'][1e-2] == 3000 and first['hits'][1e-4] == 6000 and first['hits'][1e-10] == 13000, first['hits']
    assert first['solved'] and not second['solved'] and second['hits'][1e-2] == 3000 and second['hits'][1e-4] is None
    row = summary[0]
    assert row['solve_rate'] == 0.5 and row['ert_1e-10'] == (13000 + 20000) / 1 and row['ert_0.01'] == (3000 + 3000) / 2, row
    assert math.isclose(row['median_log10_test'], (-10 + -3) / 2), row
    final_ecdf = [r for r in ecdf if r['evaluations'] == 20000][0]
    assert math.isclose(final_ecdf['share_reached'], (5 + 1) / 10), final_ecdf
    assert floor_log(None) == LOG_CAP and floor_log(math.inf) == LOG_CAP and floor_log(1e-30) == -10
    assert floor_log(10) == 1 and floor_log(1e5) == LOG_CAP and floor_log(10) < floor_log(None), 'a failure never beats a valid fit'
    parsed = json.loads('{"TestNmse": "Infinity", "Expression": "x"}', object_hook=named_floats)
    assert parsed['TestNmse'] == math.inf and parsed['Expression'] == 'x', parsed
    # Statistics against hand-computed values.
    n, w_plus, p = wilcoxon([1, 2, 3, 4, 5, 6])
    assert n == 6 and w_plus == 21 and math.isclose(p, 2 / 64), (n, w_plus, p)
    def brute(differences):
        """Exact Pratt signed-rank p by enumerating all sign assignments (independent of the dynamic programme)."""
        import itertools
        values = list(differences)
        ranked = sorted(abs(v) for v in values)
        rank_of = {a: (ranked.index(a) + 1 + len(ranked) - ranked[::-1].index(a)) / 2 for a in set(ranked)}   # mean of first and last position
        nonzero = [rank_of[abs(v)] for v in values if v != 0]
        observed = sum(rank_of[abs(v)] for v in values if v > 0)
        sums = [sum(r for r, sign in zip(nonzero, signs) if sign) for signs in itertools.product([0, 1], repeat=len(nonzero))]
        return min(1.0, 2 * min(sum(t <= observed for t in sums), sum(t >= observed for t in sums)) / len(sums))
    for sample in ([0, -1, 2, -3, 4, 5], [1, 1, 1, 1, 1], [0, 0, 1, 1, -1, 2, 2, 3], [-0.5, 0.5, 1, -2, 2, 2, 0]):
        assert math.isclose(wilcoxon(sample)[2], brute(sample)), (sample, wilcoxon(sample), brute(sample))
    assert math.isclose(wilcoxon([1, 1, 1, 1, 1])[2], 2 / 32), 'five tied improvements: exact p = 0.0625, not significant'
    assert wilcoxon([0, -1, 2, -3, 4, 5])[:2] == (5, 14.0), 'Pratt: the zero takes rank 1, the others ranks 2..6'
    assert math.isclose(mcnemar(8, 1), 2 * (1 + 9) / 512) and mcnemar(0, 0) == 1.0
    assert [round(x, 10) for x in holm([0.01, 0.04, 0.03])] == [0.03, 0.06, 0.06]
    low, high = ert_interval(jobs, 1e-10)
    # Resamples hold rep 1 twice (ERT 13,000), both reps (33,000) or rep 2 twice (no success: inf).
    assert low == 13000 and high == math.inf, (low, high)
    # A failed search without a result counts as unsolved and is charged the full budget.
    lost = failed_job({'Algorithm': 'M', 'Instance': 'T', 'Repetition': 3, 'T': 2, 'E': 10000},
                      {'Search': {'Exit': 'configurator_invalid', 'RemainingEvaluations': 20000, 'InvalidReplies': 3}})
    jobs3, summary3, _, _ = analyse([rep1, rep2], [lost])
    assert summary3[0]['repetitions'] == 3 and math.isclose(summary3[0]['solve_rate'], 1 / 3) and lost['final_log10'] == LOG_CAP
    assert summary3[0]['ert_1e-10'] == 13000 + 20000 + 20000 and lost['search_exit'] == 'configurator_invalid', summary3[0]
    try:
        analyse([rep1, rep1])
        raise AssertionError('a duplicate job must be rejected')
    except ValueError:
        pass
    # Task-weighted contrasts and interactions on hand-made jobs: two tasks, three repetitions on task X, one on task Y.
    def made(method, task, rep, value, solved=False):
        return {'method': method, 'task': task, 'repetition': rep, 'final_log10': value, 'solved': solved, 'noisy': False}
    toy = [made('A', 'X', r, -6) for r in (1, 2, 3)] + [made('B', 'X', r, -3) for r in (1, 2, 3)] + \
          [made('A', 'Y', 1, -2), made('B', 'Y', 1, -1)] + [made('C', 'X', r, -4) for r in (1, 2)] + [made('D', 'X', r, -4) for r in (1, 2)]
    plain, = contrasts(toy, ['F:A~B'], resamples=200)
    assert plain['tasks'] == 2 and plain['pairs'] == 4 and plain['mean_task_difference'] == (-3 + -1) / 2, plain   # tasks weigh equally
    assert plain['tasks_a_better'] == 2 and plain['ci_low'] == plain['ci_high'] == -2, plain   # tasks fixed; no spread within tasks
    inter, = contrasts(toy, ['F:A~B//C~D'], resamples=200)
    assert inter['tasks'] == 1 and inter['pairs'] == 2 and inter['mean_task_difference'] == -3, inter       # common repetitions 1-2
    # The failure and retry policy on synthetic protocols (scripts/attempts.py).
    import tempfile
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import attempts
    with tempfile.TemporaryDirectory() as folder:
        started = {'Algorithm': 'M', 'Instance': 'T', 'Repetition': 1, 'T': 2, 'E': 10000}
        def protocol(job, attempt, *records):
            with open(os.path.join(folder, f'{job}__attempt={attempt}.protocol.jsonl'), 'w', encoding='utf-8') as handle:
                for i, (kind, data) in enumerate(records):
                    handle.write(json.dumps({'Utc': f'2026-11-05T0{attempt}:00:0{i}Z', 'Type': kind, 'Data': data}) + '\n')
        def result(job):
            open(os.path.join(folder, job + '.result.json'), 'w').close()
        protocol('stalled', 1, ('job_started', started), ('job_stalled', {'Minutes': 60}))
        protocol('gp', 1, ('job_started', started), ('job_failed', {'Error': 'A GP trial failed after it had started', 'Search': None}))
        protocol('server', 1, ('job_started', started), ('job_failed', {'Error': 'The configurator request failed: failed:HttpRequestException'}))
        protocol('rerun', 1, ('job_started', started), ('job_failed', {'Error': 'AgentSearch evaluated no configurations'}))
        protocol('rerun', 2, ('job_started', started), ('job_completed', {})); result('rerun')
        protocol('killed', 1, ('job_started', started))
        protocol('killed', 2, ('job_started', started), ('job_completed', {})); result('killed')
        protocol('setup', 1, ('job_failed', {'Error': 'Model is not available'}))
        with open(os.path.join(folder, 'expected-jobs.txt'), 'w') as handle:
            handle.write('\n'.join(['stalled', 'gp', 'server', 'rerun', 'killed', 'setup', 'never']))
        status = {name: entry['status'] for name, entry in attempts.resolve(folder).items()}
        assert status == {'stalled': 'failure', 'gp': 'failure', 'server': 'unresolved', 'rerun': 'failure',
                          'killed': 'result', 'setup': 'unresolved', 'never': 'unresolved'}, status
    # Directories analysed together must agree: code fingerprint, server, budget, and digests of shared models.
    same = {'code': 'f1', 'server': '0.40.3', 'models': {'m4': 'd4'}, 'budget': (30, 30000, 5)}
    assert attempts.cross_directory_problems({'a': same, 'b': dict(same, models={'m4': 'd4', 'm9': 'd9'})}) == []
    assert len(attempts.cross_directory_problems({'a': same, 'b': dict(same, code='f2'), 'c': dict(same, models={'m4': 'dX'})})) == 2
    print('self-test passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directories', nargs='*')
    parser.add_argument('--out', default='./analysis')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--compare', action='append', default=[],
                        help='a family of contrasts, "NAME:A~B,A~C//B~D"; repeat for several families (Holm per family)')
    parser.add_argument('--allow-incomplete', action='store_true', help='analyse although jobs are unresolved (a look, not inference)')
    parser.add_argument('--allow-mixed', action='store_true', help='analyse results of different code or models (development only)')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.directories:
        parser.error('give at least one result directory')
    results, failed, notes = load_studies(args.directories, args.allow_incomplete, args.allow_mixed)
    for note in notes:
        print(note)
    if not results and not failed:
        sys.exit('no results and no counted failures found')
    for job in failed:
        print(f"  failure counted as unsolved: {job['method']} {job['task']} rep {job['repetition']} ({job['search_exit'] or 'failed or stalled'})")
    jobs, summary, ecdf, anytime = analyse(results, failed)
    os.makedirs(args.out, exist_ok=True)
    job_rows = [{key: value for key, value in job.items() if key not in ('path', 'hits', 'targets')} |
                {f'evaluations_to_{target:g}': job['hits'][target] for target in job['targets']} for job in jobs]
    write_csv(os.path.join(args.out, 'jobs.csv'), job_rows)
    write_csv(os.path.join(args.out, 'summary.csv'), summary)
    write_csv(os.path.join(args.out, 'ecdf.csv'), ecdf)
    write_csv(os.path.join(args.out, 'anytime.csv'), anytime)
    if args.compare:
        compared = contrasts(jobs, args.compare)
        write_csv(os.path.join(args.out, 'contrasts.csv'), compared)
        for row in compared:
            mean = row['mean_task_difference']
            print(f"  [{row['family']}] {row['contrast']}: {row['tasks']} tasks, {row['pairs']} pairs ({row['unpaired_jobs']} unpaired jobs), "
                  f"task-weighted mean log10 difference {'' if mean == '' else round(mean, 3)} "
                  f"[{'' if row['ci_low'] == '' else round(row['ci_low'], 3)}, {'' if row['ci_high'] == '' else round(row['ci_high'], 3)}], "
                  f"p {row['p']:.3g}, p_holm {row['p_holm']:.3g}")
    print(f"{len(jobs)} jobs, {len(summary)} method-task groups -> {args.out}")
    for row in summary:
        erts = ', '.join(f"ERT<{key[4:]}={value:,.0f}" if math.isfinite(value) else f"ERT<{key[4:]}=inf"
                         for key, value in row.items() if key.startswith('ert_') and not key.endswith(('_ci_low', '_ci_high')))
        solve = f"solve rate {row['solve_rate']:.0%}; " if row['solve_rate'] != '' else ''
        print(f"  {row['method']:45} {row['task']:10} n={row['repetitions']:2}  median log10 test {row['median_log10_test']:6.2f}; "
              f"{solve}budget used {row['mean_budget_used']:.0%}; {erts}")


if __name__ == '__main__':
    main()
