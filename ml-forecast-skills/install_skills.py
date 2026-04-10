#!/usr/bin/env python3
"""
install_skills.py
Instala as skills do ML-Forecast-Seg no diretório do projeto.

Uso:
    python install_skills.py
    python install_skills.py --target /caminho/para/ML-Forecast-Seg
"""
import os
import shutil
import argparse


def install(target_dir: str):
    src = os.path.join(os.path.dirname(__file__), ".claude")
    dst = os.path.join(target_dir, ".claude")

    if os.path.exists(dst):
        print(f"⚠️  Diretório .claude já existe em {target_dir}")
        resp = input("Sobrescrever? (s/N): ").strip().lower()
        if resp != 's':
            print("Instalação cancelada.")
            return

    shutil.copytree(src, dst, dirs_exist_ok=True)

    # Cria CLAUDE.md na raiz se não existir
    claude_md = os.path.join(target_dir, "CLAUDE.md")
    if not os.path.exists(claude_md):
        with open(claude_md, 'w', encoding='utf-8') as f:
            f.write(CLAUDE_MD_CONTENT)
        print(f"✅ CLAUDE.md criado em {target_dir}")

    skills = [d for d in os.listdir(os.path.join(dst, "skills"))
              if os.path.isdir(os.path.join(dst, "skills", d))]

    print(f"\n✅ Skills instaladas em {dst}/skills/:")
    for s in sorted(skills):
        print(f"   • {s}")
    print(f"\nTotal: {len(skills)} skills")
    print("\n🚀 Pronto! Abra o projeto no Claude Code e as skills serão carregadas automaticamente.")


CLAUDE_MD_CONTENT = """\
# ML-Forecast-Seg — Instruções para Claude Code

## SEMPRE leia primeiro

Antes de qualquer tarefa neste projeto, leia:
`.claude/skills/00_project_context/SKILL.md`

Ela define stack, regras invioláveis e aponta para a skill correta
de acordo com a tarefa solicitada.

## Mapa rápido de skills

| Tarefa                          | Skill                          |
|---------------------------------|--------------------------------|
| Limpar/preparar dados           | `01_data_preprocessing`        |
| Criar features temporais        | `02_feature_engineering`       |
| Detectar outliers               | `03_outlier_anomaly`           |
| Treinar XGBoost                 | `04_xgboost_forecast`          |
| Treinar LightGBM                | `05_lightgbm_forecast`         |
| Avaliar modelo / métricas       | `06_evaluation_metrics`        |
| Organizar pipeline / scripts    | `07_pipeline_structure`        |

## Regras invioláveis

1. Nunca usar dados futuros no treino (lag mínimo = horizon)
2. Validação sempre com `TimeSeriesSplit(gap=HORIZON)`
3. `random_state=42` em todo lugar
4. Comparar sempre com baseline naive antes de reportar resultado
5. Salvar modelos com timestamp e métricas no nome

## Executar pipeline completo

```bash
python run_pipeline.py --model both
```
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default=os.getcwd(),
                        help='Diretório raiz do projeto (default: diretório atual)')
    args = parser.parse_args()
    install(args.target)
