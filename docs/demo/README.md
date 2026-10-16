# Demo page

`mnemo_demo_template.html` is the progress-demo page. Its data comes from a finished Kaggle run (the unzipped
`mnemo_results.zip`) and is filled into the template:

```bash
python -m evaluation.export_demo --results data/kaggle_results/v3_final \
    --annotations data/demo/EPIC_100_train.csv --classes-dir data/demo --out /tmp/demo_data.json
python -m evaluation.make_figures --results data/kaggle_results/v3_final --out docs/figures
python - <<'PY'
t = open("docs/demo/mnemo_demo_template.html").read()
d = open("/tmp/demo_data.json").read().replace("</", "<\\/")
open("/tmp/mnemo_demo.html", "w").write(t.replace("__DATA__", d))
PY
```
Put the three PNGs from `docs/figures/` next to the built page.
