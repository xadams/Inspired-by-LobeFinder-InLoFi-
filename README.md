This script quantifies individual lobes of dicot leaves, inspired by this [paper](https://academic.oup.com/plphys/article/171/4/2331/6115282) with key differences of quantifying individual lobes.

![alt text](https://github.com/xadams/LopePlotter/blob/master/Screen%20Shot%202022-02-02%20at%205.48.24%20PM.png?raw=true)

## Usage

The script is used in the following way:

```python
python quantify_lobes.py -l listfile -o outputfile
```

Additional information can be provided with the help tag:

```python
python quantify_lobes -h
```

To generate a list file of all text files in a given directory, you can use the following bash command:

```bash
ls *txt > listfile
```

To copy the script to a different directory, use a symbolic link so that it will update with each git pull.

```bash
ln -s path/to/quantify_lobes desired/location
```
