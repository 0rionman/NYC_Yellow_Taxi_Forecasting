"""Entry points shared by Colab and local execution."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import pandas as pd
from .core import Config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','eda','train'])
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--root',default='.')
    args=parser.parse_args()
    config=Config(**json.loads(Path(args.config).read_text(encoding='utf-8')))
    root=Path(args.root)
    artifacts=root/'artifacts'
    artifacts.mkdir(parents=True,exist_ok=True)
    if args.command=='prepare':
        (artifacts/'run_status.json').write_text(json.dumps({'status':'preparing'}),encoding='utf-8')
        from .data import prepare
        prepare(root,config)
        return
    panel=pd.read_parquet(artifacts/'panel.parquet')
    from .reporting import eda,final_report
    if args.command=='eda':
        eda(panel,config,artifacts)
        print((artifacts/'EDA_REPORT.md').read_text(encoding='utf-8'))
        return
    from .experiment import run_experiment
    (artifacts/'run_status.json').write_text(json.dumps({'status':'training'}),encoding='utf-8')
    # Freeze the exact config before any evaluation of the new holdout.
    (artifacts/'frozen_protocol.json').write_text(json.dumps(asdict(config),indent=2,ensure_ascii=False),encoding='utf-8')
    table,leaderboard,results,model=run_experiment(panel,config,artifacts)
    final_report(table,leaderboard,results,model,config,artifacts)
    (artifacts/'run_status.json').write_text(json.dumps({'status':'complete'}),encoding='utf-8')
    print(results.to_string(index=False))
    print((artifacts/'REPORT.md').read_text(encoding='utf-8'))


if __name__=='__main__':
    main()
