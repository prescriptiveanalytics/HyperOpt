"""Content fingerprint of the code that determines results, as recorded in every result (Provenance.CodeFingerprint).

It identifies the code by content, not by git history, so a public copy of the repository with other commits can show
that it holds the code that produced the results: run this script in the copy and compare the value with the results
(or with study-identity.json). It must give the same value as HyperOp.Algorithms/Logging/CodeFingerprint.cs:

  every file below HyperOp.Algorithms, HyperOp.Experiments, data and docker, except in directories named bin, obj,
  .artifacts or Properties; its path relative to the repository with '/' separators; its bytes with every CR LF
  replaced by LF; the files in ordinal order of their paths; SHA-256 over the lines "<path>\\0<SHA-256 of content>\\n".

    python scripts/code_fingerprint.py [repository root] [--files]

Standard library only. The analysis scripts are not part of it: they are deterministic functions of the results.
"""
import hashlib
import os
import sys

ROOTS = ['HyperOp.Algorithms', 'HyperOp.Experiments', 'data', 'docker']
EXCLUDED = {'bin', 'obj', '.artifacts', 'Properties'}


def files(root):
    found = []
    for name in ROOTS:
        base = os.path.join(root, name)
        for directory, subdirectories, names in os.walk(base):
            for file in names:
                relative = os.path.relpath(os.path.join(directory, file), root).replace(os.sep, '/')
                if not any(part in EXCLUDED for part in relative.split('/')[:-1]):
                    found.append(relative)
    # Ordinal order (UTF-16 code units in .NET; identical to code-point order for these ASCII paths).
    return sorted(found)


def fingerprint(root):
    outer = hashlib.sha256()
    for relative in files(root):
        with open(os.path.join(root, relative), 'rb') as handle:
            content = handle.read().replace(b'\r\n', b'\n')
        outer.update(f'{relative}\0{hashlib.sha256(content).hexdigest()}\n'.encode('utf-8'))
    return outer.hexdigest()


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--files']
    root = args[0] if args else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if '--files' in sys.argv:
        print('\n'.join(files(root)))
    print(fingerprint(root))
