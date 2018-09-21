# Edits to The Original Repository

This repository is a fork of the original project [dominiek/word2vec-explorer](https://github.com/dominiek/word2vec-explorer). We removed the [Cherrypy](https://cherrypy.org/) web framework and added [Flask](flask.pocoo.org/) to the project. We also replaced the [tsne](https://pypi.org/project/tsne/) library we use the t-SNE implementation provided by the [Scikit-Learn](http://scikit-learn.org/stable/) framework. 

The server now accepts in input a general pickled object that contains the embeddings. We also provide a script to convert gensim embeddings to their pickled version.


### Setup

To install all Python depenencies:

```bash
pip install -r requirements.txt
```

### Convert gensim embeddings to embedding obj

create a directory to manage converted models

```bash
mkdir model_files
```

Run the script that converts the embedding model into a pickled object

```python
python3 convert_gensim_word2vec_model_to_embedding_file.py word2vec_file_path
```

### Run

```python
python3 server.py new_embedding_object_file
```

# Word2Vec Explorer (Original guide, for sake of completeness)

This tool helps you visualize, query and explore Word2Vec models. Word2Vec is a deep learning technique that feeds massive amounts of text into a shallow neural net which can then be used to solve a variety of NLP and ML problems.

Word2Vec Explorer uses [Gensim](https://github.com/piskvorky/gensim) to list and compare vectors and it uses [t-SNE](https://github.com/danielfrg/tsne) to visualize a dimensional reduction of the vector space. [Scikit-Learn](http://scikit-learn.org/stable/) is used for K-Means clustering.

The UI is built using [React](https://facebook.github.io/react/), [Babel](https://babeljs.io/), [Browserify](http://browserify.org/), [StandardJS](http://standardjs.com/), [D3](http://d3js.org) and [Three.js](http://threejs.org).

![TSNE 10K](https://raw.githubusercontent.com/dominiek/word2vec-explorer/master/public/screenshots/tsne-10k.png?token=AABIgK4MtRPmjZz5pWmdlLwlZtC8-hBqks5W6aenwA%3D%3D)

![TSNE Labels](https://raw.githubusercontent.com/dominiek/word2vec-explorer/master/public/screenshots/tsne-labels.png?token=AABIgEjLDw4w_O1CWfaLvQwSoroJUHhDks5W6ahRwA%3D%3D)

![Vector Comparisons](https://raw.githubusercontent.com/dominiek/word2vec-explorer/master/public/screenshots/vector-comparison.png?token=AABIgEz7KfbtSuys4yjTW9Un3QoQ4BJLks5W6ahrwA%3D%3D)

### Setup

To install all Python depenencies:

```bash
pip install -r requirements.txt
```

### Usage

Load the explorer with a Word2Vec model:

```bash
./explore GoogleNews-vectors-negative300.bin
```

Now point your browser at [localhost:8080](http://localhost:8080/) to load the explorer!

### Obtaining Pre-Trained Models

A classic example of Word2Vec is the Google News model trained on 600M sentences: [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

[More pre-trained models]](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models)

### Development

In order to make changes to the user interface you will need some NPM dependencies:

```bash
npm install
npm start
```

The command `npm start` will automatically transpile and bundle any code changes in the `ui/` folder. All backend code can be found in `explorer.py` and `./explore`.

Before submitting code changes make sure all code is compliant with StandardJS as well as Pep8:

```bash
standard
pep8 --max-line-length=100 *.py explore
```

### Todo

- 3D GPU/WebGL view (on branch `3d`)
- Make sure axes stay when zooming/panning scatterplot
- Autocomplete in query interface
- Look into supporting other high dimensional data models (go beyond word vectors)
- Drill-down of vector that shows real distance between neighbors
- Improved sample rated view that takes into account term counts and connectedness
