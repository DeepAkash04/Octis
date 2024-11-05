from octis.models.model import AbstractModel
import numpy as np
import lda
import logging
import gensim.corpora as corpora
from sklearn.feature_extraction.text import CountVectorizer
import octis.configuration.citations as citations
import octis.configuration.defaults as defaults

class LDA_gibbs(AbstractModel):

    """
        Latent Dirichlet allocation using collapsed Gibbs sampling

        Parameters
        ----------
        num_topics : int
            Number of topics

        n_iter : int, default 2000
            Number of sampling iterations

        alpha : float, default 0.1
            Dirichlet parameter for distribution over topics

        eta : float, default 0.01
            Dirichlet parameter for distribution over words

        random_state : int or RandomState, optional
            The generator used for the initial topics.


        Attributes
        ----------
        `components_` : array, shape = [num_topics, n_features]
            Point estimate of the topic-word distributions (Phi in literature)
        `topic_word_` :
            Alias for `components_`
        `nzw_` : array, shape = [num_topics, n_features]
            Matrix of counts recording topic-word assignments in final iteration.
        `ndz_` : array, shape = [n_samples, num_topics]
            Matrix of counts recording document-topic assignments in final iteration.
        `doc_topic_` : array, shape = [n_samples, n_features]
            Point estimate of the document-topic distributions (Theta in literature)
        `nz_` : array, shape = [num_topics]
            Array of topic assignment counts in final iteration.
    """
        
    id2word = None
    id_corpus = None
    use_partitions = False
    update_with_test = False

    def __init__(self, num_topics=100, n_iter=2000, alpha=0.1, eta=0.001, random_state=None, refresh=10):

        super().__init__()
        self.hyperparameters = dict()
        self.hyperparameters["n_topics"] = num_topics
        self.hyperparameters["n_iter"] = n_iter
        self.hyperparameters["alpha"] = alpha
        self.hyperparameters["eta"] = eta
        self.hyperparameters["random_state"] = random_state
        self.hyperparameters["refresh"] = refresh

        # if alpha <= 0 or eta <= 0:
        #     raise ValueError("alpha and eta must be greater than zero")

        # # random numbers that are reused
        # rng = lda.utils.check_random_state(random_state)
        # self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates

        # # configure console logging if not already configured
        # if len(logger.handlers) == 1 and isinstance(logger.handlers[0], logging.NullHandler):
        #     logging.basicConfig(level=logging.INFO)

    def info(self):
        """
            Returns model informations
        """
        return {
            "citation": citations.models_LDA_gibbs,
            "name": "LDA, Gibbs Sampling"
        }
    
    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.LDA_hyperparameters_info
    
    def train_model(self, dataset, hyperparameters=None, top_words=10):
        """
        Train the model and return output

        Parameters
        ----------
        dataset : dataset to use to build the model
        hyperparams : hyperparameters to build the model
        top_words : if greater than 0 returns the most significant words for
                    each topic in the output (Default True)
        Returns
        -------
        result : dictionary with up to 3 entries,
                 'topics', 'topic-word-matrix' and
                 'topic-document-matrix'
        """
        # if hyperparameters is None:
        #     hyperparameters = {}

        # if self.use_partitions:
        #     train_corpus, test_corpus = dataset.get_partitioned_corpus(
        #         use_validation=False)
        # else:
        #     train_corpus = dataset.get_corpus()

        # if self.id2word is None:
        #     self.id2word = corpora.Dictionary(train_corpus)

        # if self.id_corpus is None:
        #     self.id_corpus = [self.id2word.doc2bow(document) for document in train_corpus]

        # self.hyperparameters.update(hyperparameters)

        # # print(f'self.hp: {self.hyperparameters} \nhp: {hyperparameters}') #remove_edit

        # # Create a bag-of-words representation of the corpus
        # vectorizer = CountVectorizer(vocabulary=self.id2word.token2id)
        # X = vectorizer.fit_transform([' '.join(text) for text in train_corpus]).toarray()




        # model = lda.LDA(**self.hyperparameters)  
        # topic_document_matrix = model.fit_transform(X)
        # topic_word_matrix = model.topic_word_

        # result = {}
        # # result['id2word'] = self.id2word
        # # result['id_corpus'] = self.id_corpus
        # result['X'] = X
        # result["topic-document-matrix"] = topic_document_matrix
        # result["topic-word-matrix"] = topic_word_matrix

        # if top_words > 0:
        #     result["topics"] = self.get_topics(topic_word_matrix, top_words)

        # return result
    
        if hyperparameters is None:
            hyperparameters = {}

        # Obtain corpus
        if self.use_partitions:
            train_corpus, test_corpus = dataset.get_partitioned_corpus(use_validation=False)
        else:
            train_corpus = dataset.get_corpus()

        # Build the dictionary if not already created
        if self.id2word is None:
            self.id2word = corpora.Dictionary(train_corpus)

        # Prepare the corpus in BoW format (sparse tuples)
        if self.id_corpus is None:
            self.id_corpus = [self.id2word.doc2bow(document) for document in train_corpus]

        # Update hyperparameters if provided
        self.hyperparameters.update(hyperparameters)

        # Create a dense bag-of-words representation of the corpus
        vectorizer = CountVectorizer(vocabulary=self.id2word.token2id)
        X = vectorizer.fit_transform([' '.join(text) for text in train_corpus]).toarray()  # Dense array

        # Train the LDA model
        model = lda.LDA(**self.hyperparameters)
        topic_document_matrix = model.fit_transform(X)
        topic_word_matrix = model.topic_word_

        # Store results
        result = {}
        result['X'] = X  # Store dense X
        result["topic-document-matrix"] = topic_document_matrix
        result["topic-word-matrix"] = topic_word_matrix

        if top_words > 0:
            result["topics"] = self.get_topics(topic_word_matrix, top_words)

        return result
    
    def get_topics(self, topic_word_matrix, top_words):
        topic_list = []
        for topic in topic_word_matrix:
            words_list = sorted(list(enumerate(topic)), key=lambda x: x[1], reverse=True)
            topk = [tup[0] for tup in words_list[0:top_words]]
            topic_list.append([self.id2word[i] for i in topk])
        return topic_list
    