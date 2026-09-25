"""Static publication guard. No network, model inference or evaluation payloads."""
import ast
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
errors=[]
exceptions=('reproducibility/source-provenance.json','patches/observation-likelihood/',
            'src/robocup_perception/localization/synthetic_heading.py')
for path in ROOT.rglob('*'):
    rel=path.relative_to(ROOT).as_posix()
    if not path.is_file() or any(p in {'.git','__pycache__','.pytest_cache','build','artifacts','.venv'} for p in path.parts):continue
    if path.suffix in {'.pt','.pth','.onnx','.bin','.sqlite3','.npy','.npz'}:errors.append(f'Payload in Git workspace: {rel}')
    if path.suffix not in {'.md','.py','.json','.toml','.yml','.yaml','.cpp','.patch','.txt','.cff'}:continue
    text=path.read_text(encoding='utf-8-sig')
    if path.suffix=='.py':
        try:ast.parse(text)
        except SyntaxError as e:errors.append(f'Syntax: {rel}: {e}')
    if not rel.startswith(exceptions) and re.search(r'\b[Pp]rompt\d+',text):errors.append(f'Internal study label: {rel}')
    if rel!='scripts/audit_publication.py' and re.search(r'(?:C:[/\\]Users[/\\]|/home/geladeira|192\.168\.\d+\.\d+)',text):errors.append(f'Private machine path/address: {rel}')
    if path.suffix=='.md':
        for link in re.findall(r'\]\(([^)]+)\)',text):
            target=link.split('#')[0]
            if not target or re.match(r'[a-z]+:',target):continue
            if not (path.parent/unquote(target)).exists():errors.append(f'Broken link: {rel} -> {link}')
mapping=ROOT/'reproducibility/enseirb-references.json'
if mapping.exists():
    refs=json.loads(mapping.read_text())['references']
    if [r['enseirb_reference'] for r in refs]!=list(range(6,22)):errors.append('Invalid sixteen-reference mapping')
    for row in refs:
        if not (ROOT/row['path']).is_file():errors.append('Missing report: '+row['path'])
print(json.dumps({'passed':not errors,'errors':errors},indent=2))
sys.exit(bool(errors))
