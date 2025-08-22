import nbformat
from nbconvert import PythonExporter

def ipynb_to_py(ipynb_file, py_file=None):
    with open(ipynb_file, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(nb)
    
    if not py_file:
        py_file = ipynb_file.replace(".ipynb", ".py")
    
    with open(py_file, "w", encoding="utf-8") as f:
        f.write(script)
    
    print(f"Converted: {ipynb_file} -> {py_file}")

# Example usage:
ipynb_to_py("Assignment_2_CAI_PART_1.ipynb")
