from octis.models.model import AbstractModel
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import octis.configuration.defaults as defaults

class SVD_scikit(AbstractModel):
    
    def __init__(
        self, n_components=100, algorithm='randomized', n_iter=5, n_oversamples=10,
            power_iteration_normalizer='auto', random_state=None,use_partitions=False):

        super().__init__()
        self.hyperparameters["n_components"] = n_components
        self.hyperparameters["algorithm"] = algorithm
        self.hyperparameters["n_iter"] = n_iter
        self.hyperparameters["n_oversamples"] = n_oversamples
        self.hyperparameters["power_iteration_normalizer"] = power_iteration_normalizer
        self.hyperparameters["random_state"]=random_state
        self.use_partitions = use_partitions
        
        self.id2word = None
        self.id_corpus = None
        self.update_with_test = False
    
    def hyperparameters_info(self):
        """
        Returns hyperparameters informations
        """
        return defaults.SVD_scikit_hyperparameters_info

    def partitioning(self, use_partitions, update_with_test=False):
            """
            Handle the partitioning system to use and reset the model to perform
            new evaluations

            Parameters
            ----------
            use_partitions: True if train/set partitioning is needed, False
                            otherwise
            update_with_test: True if the model should be updated with the test set,
                            False otherwise
            """
            self.use_partitions = use_partitions
            self.update_with_test = update_with_test
            self.id2word = None
            self.id_corpus = None


    def train_model(self, dataset, hyperparameters=None, top_words=20):

        if hyperparameters is None:
            hyperparameters = {}

        if self.id2word is None or self.id_corpus is None:
            vectorizer = TfidfVectorizer(min_df=0.0, token_pattern=r"(?u)\b[\w|\-]+\b", vocabulary=dataset.get_vocabulary())

            if self.use_partitions:
                partition = dataset.get_partitioned_corpus(use_validation=False)
                corpus = partition[0]
            else:
                corpus = dataset.get_corpus()

            real_corpus = [" ".join(document) for document in corpus]
            X = vectorizer.fit_transform(real_corpus)

            self.id2word = {i: k for i, k in enumerate(vectorizer.get_feature_names_out())}
            if self.use_partitions:
                test_corpus = []
                for document in partition[1]:
                    test_corpus.append(" ".join(document))
                Y = vectorizer.transform(test_corpus)
                self.id_corpus = X
                self.new_corpus = Y
            else:
                self.id_corpus = X 

        #hyperparameters["corpus"] = self.id_corpus
        #hyperparameters["id2word"] = self.id2word
        self.hyperparameters.update(hyperparameters)
        
        # model = TruncatedSVD(
        #     n_components=self.hyperparameters["n_components"],
        #     algorithm=self.hyperparameters["algorithm"],
        #     n_iter=self.hyperparameters["n_iter"],
        #     n_oversamples=self.hyperparameters["n_oversamples"],
        #     power_iteration_normalizer=self.hyperparameters["power_iteration_normalizer"],
        #     random_state=self.hyperparameters["random_state"])

        # W = model.fit_transform(self.id_corpus)
        # #W = W / W.sum(axis=1, keepdims=True)
        # H = model.components_
        # #H = H / H.sum(axis=1, keepdims=True)

        # result = {}

        # result["topic-document-matrix"] = np.array(W)
        # result["topic-word-matrix"] = H

        model = TruncatedSVD(
        n_components=self.hyperparameters["n_components"], 
        algorithm=self.hyperparameters["algorithm"], 
        n_iter=self.hyperparameters["n_iter"],
        n_oversamples=self.hyperparameters["n_oversamples"],
        power_iteration_normalizer=self.hyperparameters["power_iteration_normalizer"],
        random_state=self.hyperparameters["random_state"] )

        W = model.fit_transform(self.id_corpus)
        H = model.components_

        result = {}
        result["topic-document-matrix"] = np.array(W)  
        result["topic-word-matrix"] = H

        if top_words > 0:
            result["topics"] = self.get_topics(H, top_words)

        if self.use_partitions:
            if self.update_with_test:
                # NOT IMPLEMENTED YET

                result["test-topic-word-matrix"] = W

                if top_words > 0:
                    result["test-topics"] = self.get_topics(W, top_words)

                result["test-topic-document-matrix"] = H

            else:
                result["test-topic-document-matrix"] = model.transform(self.new_corpus).T

        return result

    def get_topics(self, H, top_words):
        topic_list = []
        for topic in H:
            words_list = sorted(list(enumerate(topic)), key=lambda x: x[1], reverse=True)
            topk = [tup[0] for tup in words_list[0:top_words]]
            topic_list.append([self.id2word[i] for i in topk])
        return topic_list