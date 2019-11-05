import spacy
from .f1_squad import f1_score
from .qa_models import QA_Bert


class QG_masked:
    """
    Cloze style Question Generator based on spacy named entity recognition
    """

    def __init__(self,
                 spacy_model="en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)

    def get_questions(self, text_input):
        """
        Generate a list of questions on a text
        Args:
          text_input: a string
        Returns:
          a list of question
        """
        masked_questions = []
        asws = []

        for sent in self.nlp(text_input).sents:
            for ent in sent.ents:
                id_start = ent.start_char - sent.start_char
                id_end = ent.start_char - sent.start_char + len(ent.text)
                masked_question = sent.text[:id_start] + \
                    "MASKED" + sent.text[id_end:]
                masked_questions.append(masked_question)
                asws.append(ent.text)

        return masked_questions, asws


class QA_Metric:
    """
    Question Answering based metric
    """

    def __init__(self, model=None):

        if model is None:
            model = QA_Bert()
        self.model = model

    def compute(self, questions, true_asws, evaluated_text):
        """
        Calculate the QA scores for a given text we want to evaluate and a list of questions and their answers.
        Args:
          questions: a list of string
          true_asws: a list of string
          evaluated_text: a string
        Returns:
          a dict containing the probability score and the f-score
        """
        if not questions:
            return {"avg_prob": 0, "avg_fscore": 0}

        score_prob, score_f = 0, 0
        for question, true_asw in zip(questions, true_asws):

            asw_pred, prob = self.model.predict(question, evaluated_text)

            score_prob += prob
            score_f += f1_score(asw_pred, true_asw)

        return {"avg_prob": score_prob/len(questions), "avg_fscore": score_f/len(questions)}


def evaluate_corpus(srcs, gens, model=None, questionss=None, aswss=None):
    """
    Calculate the QA scores for an entire corpus.
    Args:
      srcs: a list of string (one string per document)
      gens: a list of string (one string per summary)
      model: [optional]: any model that fits the function predict in qa_models; by default BERT_QA
      questionss: [optional]: a list of list with the questions already generated for each src. If None, it will generate it.
      aswss: [optional]: a list of list with the ground truth asws for the questions (questionss). If None, it will generate it as well.
    Returns:
      a dict containing the probability score and f-score, averaged for the corpus
    """
    assert any([questionss, aswss]) == all([questionss, aswss]
                                           ), "questionss/aswss should be None if the other is None"

    # if questionss is None initialize a question generator
    if not questionss:
        question_generator = QG_masked()
    # initialize the metric with a QA model
    qa_metric = QA_Metric(model)

    global_score = {"avg_prob": 0, "avg_fscore": 0}

    for i, (src, gen) in enumerate(zip(srcs, gens)):

        # if questionss is None, generate the questions and answers else get the corrisponding ones.
        if not questionss:
            masked_questions, masked_question_asws = question_generator.get_questions(
                src)
        else:
            masked_questions, masked_question_asws = questionss[i], aswss[i]

        # compute the metric
        gen_score = qa_metric.compute(
            masked_questions, masked_question_asws, gen)
        global_score['avg_prob'] += gen_score['avg_prob']
        global_score['avg_fscore'] += gen_score['avg_fscore']

    # average it
    global_score['avg_prob'] = global_score['avg_prob'] / len(srcs)
    global_score['avg_fscore'] = global_score['avg_fscore'] / len(srcs)

    return global_score
