This folder contains Python and C++ scripts that generate input output mappings from a specified FST graph and input string. 
The required input FST graph needs to be provided as OpenFst FAR file.

Requirements:
- Python: Pynini
- C++: Thrax, OpenFst

Example Usage:


```python
python alignment.py --fst=fst.far --text="2615 Forest Av, 1 Aug 2016" --rule="tokenize_and_classify" --start=22 --end=26

Output: \
inp string: |2615 Forest Av, 1 Aug 2016| \
out string: |twenty six fifteen Forest Avenue , the first of august twenty sixteen| \
inp indices: [22:26]
out indices: [55:69] \
in: |2016| out: |twenty sixteen| 
```



Disclaimer: 

The heuristic algorithm relies on monotonous alignment and can fail in certain situations,
e.g. when word pieces are reordered by the fst.