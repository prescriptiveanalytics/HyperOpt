"""Resolution of job attempts under the study's failure and retry policy (Supplement S4).

A job is identified by its name (the result file name without .result.json; protocols are <name>__attempt=<id>.protocol.jsonl).
Each attempt is classified from its protocol:
  completed       job_completed: the job's result file holds its outcome
  algorithmic     job_failed for a reason inside the method (a failed GP trial, a search without a valid configuration) or
                  job_stalled (a hang would recur): an outcome, counted as unsolved with the full budget
  infrastructure  job_failed caused by the model server (failed request, HTTP or connection error), or a failure before
                  job_started (setup, model resolution, warm-up): retried, the attempt is discarded
  interrupted     no terminal record (stopped, killed, crashed): retried, the attempt is discarded
Attempts are ordered by their first record's time. A job resolves to
  result      a completed attempt and no earlier algorithmic failure
  failure     the first algorithmic failure (a later successful rerun is not used)
  unresolved  neither (only discarded attempts, or nothing at all for a job in expected-jobs.txt)
Standard library only.
"""
import glob
import json
import os
import re

INFRASTRUCTURE = re.compile(r'configurator request failed|HttpRequestException|No connection could be made|actively refused|'
                            r'Connection refused|connection was closed|Ollama server|not reachable|SocketException', re.I)


def job_name(protocol_path):
    return os.path.basename(protocol_path).split('__attempt=')[0]


def classify(protocol_path):
    """(first record time, status, job_started data or None, failure data or None) of one attempt."""
    first_time, started, failure, status = '', None, None, 'interrupted'
    with open(protocol_path, encoding='utf-8-sig') as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:   # a truncated last line of a killed attempt
                continue
            first_time = first_time or record.get('Utc', '')
            kind = record.get('Type')
            if kind == 'job_started':
                started = record['Data']
            elif kind == 'job_completed':
                status = 'completed'
            elif kind == 'job_stalled':
                status, failure = 'algorithmic', record.get('Data') or {}
            elif kind == 'job_failed':
                failure = record.get('Data') or {}
                status = 'infrastructure' if started is None or INFRASTRUCTURE.search(str(failure.get('Error', ''))) else 'algorithmic'
    return first_time, status, started, failure


def resolve(directory):
    """{job name: {'status', 'result_path', 'started', 'failure', 'attempts'}} for every job with a result, a protocol, or an
    entry in expected-jobs.txt."""
    jobs = {}
    for path in glob.glob(os.path.join(directory, '*.protocol.jsonl')):
        jobs.setdefault(job_name(path), []).append((*classify(path), path))
    results = {os.path.basename(p)[:-len('.result.json')]: p for p in glob.glob(os.path.join(directory, '*.result.json'))}
    expected = set()
    listed = os.path.join(directory, 'expected-jobs.txt')
    if os.path.exists(listed):
        with open(listed, encoding='utf-8-sig') as handle:
            expected = {line.strip() for line in handle if line.strip()}
    resolved = {}
    for name in set(jobs) | set(results) | expected:
        attempts = sorted(jobs.get(name, []), key=lambda a: a[0])
        entry = {'status': 'unresolved', 'result_path': None, 'started': None, 'failure': None,
                 'attempts': [a[1] for a in attempts], 'expected': name in expected or not expected}
        for time, status, started, failure, path in attempts:
            if status == 'algorithmic':
                entry.update(status='failure', started=started, failure=failure)
                break
            if status == 'completed':
                break
        if entry['status'] != 'failure' and name in results:
            entry.update(status='result', result_path=results[name])
        resolved[name] = entry
    return resolved


def summary(resolved):
    """Counts per status and the names of unresolved and unexpected jobs."""
    counts = {'result': 0, 'failure': 0, 'unresolved': 0}
    for entry in resolved.values():
        counts[entry['status']] += 1
    unresolved = sorted(n for n, e in resolved.items() if e['status'] == 'unresolved')
    unexpected = sorted(n for n, e in resolved.items() if not e['expected'] and e['status'] != 'unresolved')
    return counts, unresolved, unexpected


def load_study(directory):
    """The resolved study in one directory, the single source for the validator and the analysis.

    Returns a dict with
      results       parsed result files of jobs that resolved to a result (each with '_path' and '_name')
      failures      (name, job_started data, failure data) of jobs that resolved to a counted failure
      counts, unresolved, unexpected   as from summary()
      completeness  problems: unresolved or unexpected jobs, results without their completed protocol
      identity      problems: against study-identity.json where present (code fingerprint, T, E, K, model digest and
                    server version, for results and for the job_started record of counted failures), otherwise mutual
                    consistency; agent results without model or context records. The code is identified by its content
                    fingerprint, so a public copy with other commits matches; jobs from before the fingerprint (no value)
                    fall back to the commit and must come from a clean build.
    """
    resolved = resolve(directory)
    counts, unresolved, unexpected = summary(resolved)
    results, failures, completeness, identity = [], [], [], []
    for name, entry in sorted(resolved.items()):
        if entry['status'] == 'result':
            with open(entry['result_path'], encoding='utf-8-sig') as handle:
                result = json.load(handle)
            result['_path'], result['_name'] = entry['result_path'], name
            protocol = result.get('ProtocolFile')
            if not protocol or not os.path.exists(os.path.join(directory, protocol)) or \
                    classify(os.path.join(directory, protocol))[1] != 'completed':
                completeness.append(f'{name}: the result does not name a completed protocol of this directory')
            results.append(result)
        elif entry['status'] == 'failure':
            failures.append((name, entry['started'] or {}, entry['failure'] or {}))
    if unresolved:
        completeness.append(f'{len(unresolved)} jobs are neither a result nor a counted failure, e.g. {unresolved[0]}')
    if unexpected:
        completeness.append(f'{len(unexpected)} jobs are not in expected-jobs.txt, e.g. {unexpected[0]}')

    # One identity per job: (code, dirty, T, E, K, model, digest, server, algorithm) from a result or a failure's job_started;
    # the code is the content fingerprint, or 'commit:<hash>' for jobs from before it, whose build must then be clean.
    def code(provenance):
        provenance = provenance or {}
        if provenance.get('CodeFingerprint'):
            return provenance['CodeFingerprint'], False
        return f"commit:{provenance.get('GitCommit')}", provenance.get('GitDirty', True)
    def of_result(r):
        info = r.get('ModelInfo') or {}
        lengths = r.get('AssessmentRunLengths') or [r.get('EvaluationsPerConfiguration')]
        k = len(r.get('ConfigurationAssessmentResults') or []) // max(1, len(lengths))
        return (*code(r.get('Provenance')),
                r.get('NumConfigurations'), r.get('EvaluationsPerConfiguration'), k, info.get('Name'), info.get('Digest'),
                info.get('ServerVersion'), r.get('AlgorithmName'))
    def of_started(s):
        info = s.get('ModelInfo') or {}
        return (*code(s.get('Provenance')),
                s.get('T'), s.get('E'), s.get('K'), info.get('Name'), info.get('Digest'), info.get('ServerVersion'), s.get('Algorithm'))
    jobs = [(r['_name'], of_result(r)) for r in results] + [(name, of_started(started)) for name, started, _ in failures]
    for name, (_, _, _, _, _, model, digest, _, algorithm) in jobs:
        if algorithm == 'AgentSearch' and not digest:
            identity.append(f'{name}: an agent job without its model record')
    for r in results:
        if r.get('AlgorithmName') == 'AgentSearch' and (r.get('Timing') or {}).get('ContextLength') is None:
            identity.append(f"{r['_name']}: an agent result without its context window")
    if any(dirty for _, (_, dirty, *_rest) in jobs):
        identity.append(f'{sum(bool(dirty) for _, (_, dirty, *_r) in jobs)} jobs without a code fingerprint were built from uncommitted code')
    fittings = {json.dumps([(r.get('Provenance') or {}).get('FitterVersion'), (r.get('Provenance') or {}).get('ConstantFitting')], sort_keys=True)
                for r in results}
    if len(fittings) > 1:
        identity.append(f'{len(fittings)} constant-fitting settings')
    identity_file = os.path.join(directory, 'study-identity.json')
    if os.path.exists(identity_file):
        with open(identity_file, encoding='utf-8-sig') as handle:
            recorded = json.load(handle)
        base, digests = recorded.get('Base') or {}, recorded.get('ModelDigests') or {}
        expected = recorded.get('CodeFingerprint') or f"commit:{recorded.get('Commit')}"
        for name, (fingerprint, _, t, e, k, model, digest, server, algorithm) in jobs:
            wrong = []
            if fingerprint != expected:
                wrong.append(f'code {str(fingerprint)[:12]}')
            if (t, e, k) != (base.get('T'), base.get('E'), base.get('K')):
                wrong.append(f'T/E/K {t}/{e}/{k}')
            if model and (digest != digests.get(model) or server != recorded.get('OllamaVersion')):
                wrong.append(f'model {model} {str(digest)[:12]} on Ollama {server}')
            if wrong:
                identity.append(f"{name}: differs from study-identity.json ({', '.join(wrong)})")
    else:
        codes = {fingerprint for _, (fingerprint, *_r) in jobs}
        if len(codes) > 1 or 'commit:None' in codes:
            identity.append(f'jobs come from {len(codes)} code versions')
        per_model = {}
        for _, (*_x, model, digest, server, _a) in jobs:
            if model:
                per_model.setdefault(model, set()).add((digest, server))
        identity += [f'{m} appears with {len(v)} digest/server combinations' for m, v in per_model.items() if len(v) > 1]
    return {'results': results, 'failures': failures, 'counts': counts, 'unresolved': unresolved, 'unexpected': unexpected,
            'completeness': completeness, 'identity': identity}


def identity_signature(directory, study):
    """What must agree between directories analysed together: code fingerprint, model digests and server version, T, E, K.
    From study-identity.json where present, otherwise from the jobs' own records."""
    identity_file = os.path.join(directory, 'study-identity.json')
    if os.path.exists(identity_file):
        with open(identity_file, encoding='utf-8-sig') as handle:
            recorded = json.load(handle)
        base = recorded.get('Base') or {}
        return {'code': recorded.get('CodeFingerprint') or f"commit:{recorded.get('Commit')}", 'server': recorded.get('OllamaVersion'),
                'models': dict(sorted((recorded.get('ModelDigests') or {}).items())),
                'budget': (base.get('T'), base.get('E'), base.get('K'))}
    commits, servers, models, budgets = set(), set(), {}, set()
    for r in study['results']:
        provenance = r.get('Provenance') or {}
        commits.add(provenance.get('CodeFingerprint') or f"commit:{provenance.get('GitCommit')}")
        budgets.add((r.get('NumConfigurations'), r.get('EvaluationsPerConfiguration')))
        info = r.get('ModelInfo') or {}
        if info:
            servers.add(info.get('ServerVersion'))
            models[info.get('Name')] = info.get('Digest')
    return {'code': sorted(map(str, commits)), 'server': sorted(map(str, servers)), 'models': dict(sorted(models.items())),
            'budget': sorted(budgets)}


def cross_directory_problems(signatures):
    """Differences between the identity signatures of directories analysed together ({directory: signature}). Models only
    need the same digest where both directories use them; the budget may differ only if the analysis keeps them apart,
    which it does not, so it must agree too."""
    problems = []
    items = list(signatures.items())
    for (first, a), (other, b) in ((items[0], item) for item in items[1:]):
        for key in ('code', 'server', 'budget'):
            if a[key] != b[key]:
                problems.append(f'{other} differs from {first} in {key}: {b[key]} vs {a[key]}')
        for model in set(a['models']) & set(b['models']):
            if a['models'][model] != b['models'][model]:
                problems.append(f'{other} uses another digest of {model} than {first}')
    return problems
