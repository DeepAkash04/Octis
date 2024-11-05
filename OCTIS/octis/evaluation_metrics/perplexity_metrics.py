from octis.evaluation_metrics.metrics import AbstractMetric
from octis.dataset.dataset import Dataset
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from gensim.models import KeyedVectors
import gensim.downloader as api
import octis.configuration.citations as citations
import numpy as np
import itertools
from scipy import spatial
from sklearn.metrics import pairwise_distances
from operator import add


class Perplexity(AbstractMetric):
    def __init__(self):
        super().__init__()

    def compute_log_likelihood(self, model_output, simplified=False):
        """
        Calculate log-likelihood for the model output.

        Parameters
        ----------
        model_output : dict
            Output of the model containing 'topic-document-matrix'.

        Returns
        -------
        float
            Log-likelihood value.
        """
        if simplified:
            W = model_output["topic-document-matrix"]
            return np.sum(np.log(np.sum(W, axis=1) + 1e-10)) # Noise 1e-10 is added to avoid log(0)
        else:
            X = model_output['X']
            WH = np.dot(model_output["topic-document-matrix"] , model_output["topic-word-matrix"])
            return np.sum(X * np.log(WH + 1e-10))
    
    def compute_perplexity(self, model_output, score_by_dtm=False):
        """
        Calculate the perplexity based on the model output.

        Parameters
        ----------
        model_output : dict
            Output of the model containing 'topic-document-matrix'.

        Returns
        -------
        float
            Perplexity score.
        """
        topic_document_matrix = model_output["topic-document-matrix"]
        topic_word_matrix = model_output['topic-word-matrix']
        D = topic_document_matrix.shape[0]  # Number of documents
        T = topic_word_matrix.shape[1] # Vocab Length

        # Calculate the log-likelihood
        log_likelihood = self.compute_log_likelihood(model_output)

        # Perplexity calculation
        perplexity = np.exp(-log_likelihood / D) if score_by_dtm else np.exp(-log_likelihood / T)
        return perplexity

    def compute_aic(self, model_output):
        """
        Compute the Akaike Information Criterion (AIC).

        Parameters
        ----------
        model_output : dict
            Output of the model containing 'topic-document-matrix'.

        Returns
        -------
        float
            AIC score.
        """
        topic_document_matrix = model_output["topic-document-matrix"]
        D = topic_document_matrix.shape[0]  # Number of documents
        k = topic_document_matrix.shape[1]  # Number of topics

        log_likelihood = self.compute_log_likelihood(model_output)

        aic = 2 * k - 2 * log_likelihood
        return aic

    def compute_bic(self, model_output):
        """
        Compute the Bayesian Information Criterion (BIC).

        Parameters
        ----------
        model_output : dict
            Output of the model containing 'topic-document-matrix'.

        Returns
        -------
        float
            BIC score.
        """
        topic_document_matrix = model_output["topic-document-matrix"]
        D = topic_document_matrix.shape[0]  # Number of documents
        k = topic_document_matrix.shape[1]  # Number of topics

        log_likelihood = self.compute_log_likelihood(model_output)

        bic = k * np.log(D) - 2 * log_likelihood
        return bic

    def score(self, model_output):
        """
        Calculate and return a dictionary containing perplexity, AIC, and BIC.

        Parameters
        ----------
        model_output : dict
            Output of the model containing 'topic-document-matrix'.

        Returns
        -------
        dict
            Dictionary containing 'perplexity', 'AIC', and 'BIC'.
        """
        perplexity = self.compute_perplexity(model_output)
        aic = self.compute_aic(model_output)
        bic = self.compute_bic(model_output)

        return {
            "perplexity": perplexity,
            "AIC": aic,
            "BIC": bic
        }


def _load_default_texts():
    """
    Loads default general texts

    Returns
    -------
    result : default 20newsgroup texts
    """
    dataset = Dataset()
    dataset.fetch_dataset("20NewsGroup")
    return dataset.get_corpus()
