"""Context usage of agent jobs: the largest prompt per role and the largest prompt plus reply, per job and per model.

Reads every *.result.json below the given directories (standard library only). Results written before the context fields
existed are measured from the llm_call records of their protocol file, which lies next to the result.

    python scripts/context_usage.py results/context-check
"""
import glob
import json
import os
import sys


def load(path):
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def from_protocol(path):
    """Largest configurator prompt, controller prompt and prompt plus reply from a protocol's llm_call records."""
    usage = {'configurator': 0, 'controller': 0, 'request': 0}
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            record = json.loads(line)
            if record.get('Type') != 'llm_call':
                continue
            data = record['Data']
            prompt = data.get('InputTokens') or 0
            usage[data['Role']] = max(usage.get(data['Role'], 0), prompt)
            usage['request'] = max(usage['request'], prompt + (data.get('OutputTokens') or 0))
    return usage['configurator'], usage['controller'], usage['request']


def jobs(directories):
    for directory in directories:
        for path in sorted(glob.glob(os.path.join(directory, '**', '*.result.json'), recursive=True)):
            result = load(path)
            if result.get('AlgorithmName') != 'AgentSearch':
                continue
            timing = result.get('Timing') or {}
            if 'MaxRequestTokens' in timing:
                usage = timing['MaxConfiguratorInputTokens'], timing['MaxControllerInputTokens'], timing['MaxRequestTokens']
            else:
                protocol = os.path.join(os.path.dirname(path), result.get('ProtocolFile') or '')
                if not os.path.isfile(protocol):
                    continue
                usage = from_protocol(protocol)
            yield {
                'model': result.get('ModelName'), 'variant': result.get('Variant'), 'instance': result.get('InstanceName'),
                'T': result.get('NumConfigurations'), 'E': result.get('EvaluationsPerConfiguration'),
                'context': timing.get('ContextLength'), 'configurator': usage[0], 'controller': usage[1], 'request': usage[2],
                'near_limit': timing.get('RequestsNearContextLimit', 0),
            }


def main(directories):
    rows = list(jobs(directories))
    if not rows:
        sys.exit('no AgentSearch results found')
    print('model\tvariant\tinstance\tT\tE\tcontext\tmax configurator prompt\tmax controller prompt\tmax prompt+reply\tnear limit')
    for r in rows:
        print('\t'.join(str(r[k]) for k in ('model', 'variant', 'instance', 'T', 'E', 'context', 'configurator', 'controller',
                                            'request', 'near_limit')))
    print('\nper model: largest prompt+reply over all jobs (share of the context window)')
    for model in sorted({r['model'] for r in rows}):
        largest = max((r for r in rows if r['model'] == model), key=lambda r: r['request'])
        share = f"{largest['request'] / largest['context']:.0%}" if largest['context'] else 'unknown window'
        print(f"{model}\t{largest['request']} tokens ({share}; {largest['variant']}, {largest['instance']})")


if __name__ == '__main__':
    main(sys.argv[1:] or ['results'])
