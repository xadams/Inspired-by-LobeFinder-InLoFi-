This script quantifies individual lobes of dicot leaves, inspired by [LobeFinder](https://academic.oup.com/plphys/article/171/4/2331/6115282). The primary output is a csv file containing measurements for each lobe, with optional output visually describing how the lobes were determined from the input. One such visual is shown below:

![alt text](https://github.com/xadams/LobePlotter/blob/master/LobePlotterDiagram.png)

## Usage

The script is provided a listfile with the paths to the image files in a listfile, and is run in the following way:

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

To copy the script to a different directory for ease of use, use a symbolic link so that it will update with each git pull.

```bash
ln -s path/to/quantify_lobes desired/location
```
