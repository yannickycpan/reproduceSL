source activate pycwork
parallel --results stdout -j 11 -u python -W ignore main.py tdsl2.json {} ::: {0..3239}
