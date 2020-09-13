# Reddit Discourse about Marijuana Legalization
This study, in collaboration with [Sloman Lab](https://sites.google.com/site/slomanlab/) and [AI Lab](https://brown.edu/Research/AI/people/carsten.html) at Brown University, involves applying unsupervised and supervised machine learning methods to examine temporal trends in discourse about marijuana legalization on Reddit since 2008.

Raw and processed versions of corpus and models, as well as detailed descriptions of the procedures used in this study can be found [here](https://drive.google.com/open?id=17PjV5gPub15kSaHpw9JVP1SNpj1k3vK-).

Acknowledgment: The basis for the code in the current repository is [this](https://github.com/BabakHemmatian/Gay_Marriage_Corpus_Study]) study on same-sex marriage discourse, which was developed in collaboration with Sabina J. Sloman from Carnegie Mellon University, and Steven A. Sloman and Uriel Cohen Priva from Brown University.

# To setup conda environment
1. Make sure you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed.
2. Clone this respository and be sure to switch to the debugging branch. 
**NOTE:** The debugging branch will soon be merged with master.
3. You will have to install spacy en_core_web_sm manually. To do this first run <kbd>pip install spacy</kbd>. Once it's installed, run <kbd>python -m spacy download en</kbd>
4. You can now create an environment using the `Marijuana_Study.yml` file. In the terminal go to the directory where you cloned this repository. Now run <kbd>conda env create -f Marijuana_Study.yml</kbd>.
5. Once the environment is created, you will need to activate the environment. To do this run <kbd>conda activate Marijuana_study</kbd>
6. You might also need nltk to run the files. To install nltk, follow [this guide](https://www.nltk.org/data.html).
