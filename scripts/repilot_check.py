"""Pre-declared pass criteria of the re-pilot and completeness check of any study (Supplement S4).

Per cell (method, model, variant): jobs, failed jobs, searches ended by invalid replies, the worst controller fallback rate,
requests at 90% of the context window, evaluations left unspent, median job minutes against the estimate, largest
request in tokens. Pass criteria, all cells:
  1. no failed job and no search ended as configurator_invalid;
  2. controller fallbacks below 1% of decisions in every job;
  3. no request at 90% of the context window;
  0. completeness and identity (scripts/attempts.py load_study, the same check as the analysis): every job of
     expected-jobs.txt ends as a result or a counted failure under the failure and retry policy, each result names its
     completed protocol, and all jobs match study-identity.json (or, without it, one another);
  4. the whole budget spent and never exceeded (irace may leave one run of E unused, as irace 4.4.0 does when exactly one
     run remains after its last race);
  5. median job time per concurrent job within 1.5 times the estimate (agent cells only): the estimate is the GPU time of
     one job alone; with CONCURRENCY agent jobs at once, Ollama serves one request at a time, so each job's wall time is
     about CONCURRENCY times longer while throughput is unchanged.

Cells and failure counts come from the resolved attempts: a retried infrastructure failure is not a failed job, and a
rerun after a counted failure is not used. Exit code 1 when any criterion fails.

    python scripts/repilot_check.py results/repilot-2026-11
    python scripts/repilot_check.py --self-test
"""
import collections
import glob
import json
import os
import statistics
import sys

# Estimated minutes per job at T = 30, E = 30,000, from the pilots' service times (re-pilot pass criterion 5).
ESTIMATE = {'2b': 6.0, '4b': 9.5, '9b': 14.0}
ESTIMATE_FINAL = {'2b': 3.0, '4b': 4.0, '9b': 6.0}
ESTIMATE_RULE = {'2b': 4.0, '4b': 5.5, '9b': 8.0}
TOLERANCE = 1.5
CONCURRENCY = 4   # Run-Experiments.ps1 -MaxAgentProcesses


def size(model):
    for key in ESTIMATE:
        if f'{key}-' in (model or ''):
            return key
    return None


def estimate(model, variant):
    key = size(model)
    if key is None:
        return None
    if variant.startswith('feedback=final'):
        return ESTIMATE_FINAL[key]
    if '__controller=' in variant:
        return ESTIMATE_RULE[key]
    return ESTIMATE[key]


def load(directory):
    """Cells, counted failures per cell, and completeness and identity problems, from the resolved study."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import attempts
    study = attempts.load_study(directory)
    cells = collections.defaultdict(list)
    for r in study['results']:
        cells[(r['AlgorithmName'], r.get('ModelName') or '', r.get('Variant') or '')].append(r)
    failed = collections.Counter((started.get('Algorithm', '?'), started.get('Model') or '', started.get('Variant') or '')
                                 for _, started, _ in study['failures'])
    problems = [f'0: {p}' for p in study['completeness'] + study['identity']]
    if not study['results'] and not study['failures']:
        problems.append('0: no results and no counted failures in ' + directory)
    if not os.path.exists(os.path.join(directory, 'expected-jobs.txt')):
        print(f'Note: no expected-jobs.txt in {directory}; completeness is checked against the results only.')
    return cells, failed, problems


def main(directory):
    cells, failed, problems = load(directory)
    print('\t'.join(['method', 'model', 'variant', 'jobs', 'failed', 'invalid ends', 'max fallback rate', 'near-limit requests',
                     'max unspent', 'median minutes', 'per concurrent job', 'estimate', 'largest request']))
    for key in sorted(set(cells) | set(failed)):
        algorithm, model, variant = key
        jobs = cells.get(key, [])
        invalid = sum((r.get('Search') or {}).get('Exit') == 'configurator_invalid' for r in jobs)
        rates = [r['Timing']['ControllerFallbacks'] / r['Timing']['ControllerRequests'] for r in jobs
                 if (r['Timing'].get('ControllerRequests') or 0) > 0]
        near = sum(r['Timing'].get('RequestsNearContextLimit') or 0 for r in jobs)
        unspent = [r['NumConfigurations'] * r['EvaluationsPerConfiguration'] - sum(t['EvaluationsUsed'] for t in r['ConfigurationResults'])
                   for r in jobs]
        minutes = statistics.median(r['Timing']['TotalJobMs'] / 60000 for r in jobs) if jobs else None
        expected = estimate(model, variant) if algorithm == 'AgentSearch' else None
        largest = max((r['Timing'].get('MaxRequestTokens') or 0 for r in jobs), default=0)
        name = f'{algorithm} {model} {variant}'.strip()
        if failed[key] or invalid:
            problems.append(f'1: {name}: {failed[key]} failed, {invalid} ended by invalid replies')
        if rates and max(rates) >= 0.01:
            problems.append(f'2: {name}: fallback rate {max(rates):.1%}')
        if near:
            problems.append(f'3: {name}: {near} requests at 90% of the context window')
        # irace treats T as a maximum: its post-selection race spends a remainder of at least two runs, but a single
        # remaining run is left unused (irace 4.4.0); that one run of E evaluations is expected, more is not.
        allowed = 100 if algorithm != 'IRaceSearch' else max((r['EvaluationsPerConfiguration'] for r in jobs), default=0) + 100
        if unspent and max(unspent) >= allowed:
            problems.append(f'4: {name}: up to {max(unspent)} evaluations left unspent')
        if unspent and min(unspent) < 0:
            problems.append(f'4: {name}: a job used {-min(unspent)} evaluations more than its budget')
        if expected and minutes and minutes / CONCURRENCY > TOLERANCE * expected:
            problems.append(f'5: {name}: median {minutes:.1f} min, {minutes / CONCURRENCY:.1f} per concurrent job, against an estimate of {expected} min')
        print('\t'.join(str(v) for v in [algorithm, model, variant, len(jobs), failed[key], invalid,
                                          f'{max(rates):.2%}' if rates else '-', near, max(unspent) if unspent else '-',
                                          f'{minutes:.1f}' if minutes else '-',
                                          f'{minutes / CONCURRENCY:.1f}' if minutes and expected else '-', expected or '-', largest or '-']))
    print()
    print('PASS: all criteria met' if not problems else 'FAIL:\n  ' + '\n  '.join(problems))
    return 1 if problems else 0


def self_test():
    """The validator on synthetic studies: a retried infrastructure failure, a counted failure followed by a rerun, a
    truncated interrupted log, a commit that differs from study-identity.json, and a study of failures only."""
    import contextlib, io as text_io, tempfile
    started = {'Algorithm': 'RandomSearch', 'Instance': 'T', 'Repetition': 1, 'T': 2, 'E': 1000, 'K': 0,
               'Provenance': {'GitCommit': 'c1', 'GitDirty': False, 'CodeFingerprint': 'f1'}}
    def result(name, protocol, code='f1'):
        return {'AlgorithmName': 'RandomSearch', 'ModelName': None, 'Variant': '', 'InstanceName': 'T', 'Repetition': 1,
                'NumConfigurations': 2, 'EvaluationsPerConfiguration': 1000, 'ProtocolFile': protocol,
                'ConfigurationResults': [{'EvaluationsUsed': 1000}, {'EvaluationsUsed': 1000}], 'ConfigurationAssessmentResults': [],
                'AssessmentRunLengths': [1000, 4000], 'Provenance': {'GitCommit': 'private', 'GitDirty': True, 'CodeFingerprint': code, 'FitterVersion': 'f'},
                'Timing': {'ControllerRequests': 0, 'ControllerFallbacks': 0, 'TotalJobMs': 60000}}
    def run(build):
        with tempfile.TemporaryDirectory() as folder:
            build(folder)
            out = text_io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(folder)
            return code, out.getvalue()
    def write(folder, job, attempt, records, text=None):
        name = f'{job}__attempt={attempt}.protocol.jsonl'
        with open(os.path.join(folder, name), 'w', encoding='utf-8') as handle:
            for i, (kind, data) in enumerate(records):
                handle.write(json.dumps({'Utc': f'2026-11-05T0{attempt}:00:0{i}Z', 'Type': kind, 'Data': data}) + '\n')
            if text:
                handle.write(text)
        return name
    def finished(folder, job, attempt, code='f1', identity=None):
        protocol = write(folder, job, attempt, [('job_started', started), ('job_completed', {})])
        with open(os.path.join(folder, job + '.result.json'), 'w', encoding='utf-8') as handle:
            json.dump(result(job, protocol, code), handle)
        with open(os.path.join(folder, 'expected-jobs.txt'), 'w', encoding='utf-8') as handle:
            handle.write(job)
        if identity:
            with open(os.path.join(folder, 'study-identity.json'), 'w', encoding='utf-8') as handle:
                json.dump(identity, handle)
    identity = {'CodeFingerprint': 'f1', 'OllamaVersion': '0.40.3', 'ModelDigests': {}, 'Base': {'T': 2, 'E': 1000, 'K': 0}}
    # Server failure, then a successful retry: no failed job.
    code, out = run(lambda f: (write(f, 'job', 1, [('job_started', started), ('job_failed', {'Error': 'The configurator request failed: HttpRequestException'})]),
                               finished(f, 'job', 2, identity=identity)))
    assert code == 0 and 'PASS' in out, out
    # Counted failure, then a rerun: the failure counts and the rerun is not used.
    code, out = run(lambda f: (write(f, 'job', 1, [('job_started', started), ('job_failed', {'Error': 'A GP trial failed after it had started'})]),
                               finished(f, 'job', 2, identity=identity)))
    assert code == 1 and '1: RandomSearch' in out and '1 failed' in out, out
    # A killed attempt with a truncated last line, then a successful one: no crash, PASS.
    code, out = run(lambda f: (write(f, 'job', 1, [('job_started', started)], text='{"Utc": "2026-11-05T01:00:09Z", "Type": "chec'),
                               finished(f, 'job', 2, identity=identity)))
    assert code == 0 and 'PASS' in out, out
    # A result of other code than the study's identity (the commit and a dirty notes tree do not matter).
    code, out = run(lambda f: finished(f, 'job', 1, code='f2', identity=identity))
    assert code == 1 and 'differs from study-identity.json' in out, out
    # Failures only: counted, no spurious identity error.
    code, out = run(lambda f: (write(f, 'job', 1, [('job_started', started), ('job_stalled', {})]),
                               open(os.path.join(f, 'expected-jobs.txt'), 'w').write('job')))
    assert code == 1 and '1 failed' in out and 'code versions' not in out and 'no results' not in out, out
    print('self-test passed')


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        self_test()
        sys.exit(0)
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
