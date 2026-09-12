import ast, io, os, re
os.chdir(r"C:\Users\Lenovo\Documents\MicroVLM-PTQ-Benchmark\python")
d = os.getcwd()
print("CWD:", d)

src_e = io.open("export_quant.py", encoding="utf-8").read()
src_m = io.open("main_pipeline.py", encoding="utf-8").read()

print("---- export_quant.py convert_all def (verbatim) ----")
for i, l in enumerate(src_e.splitlines(), 1):
    if l.strip().startswith("def convert_all"):
        print(f"{i}: {l}")
for i, l in enumerate(src_e.splitlines(), 1):
    if "representative_dataset" in l:
        print(f"{i}: {l.strip()}")

print("---- main_pipeline.py convert_all call (verbatim) ----")
for i, l in enumerate(src_m.splitlines(), 1):
    if "convert_all(" in l:
        print(f"{i}: {l.strip()}")
