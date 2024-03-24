# Robot Teaching Assiantant

To run use the command below:

```
python run.py --config config.yaml
```


To compile the example cpp:

```
c++ -O3 -Wall -shared -std=c++11 -fPIC -undefined dynamic_lookup $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
```

To download the pretrained gan model [Download](https://drive.google.com/file/d/1isZ5VeZV4Sx2XI0tt7SnsvH2JjcvowLY/view?usp=sharing)
