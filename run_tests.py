from allennlp_models.pretrained import load_predictor
from checklist.test_types import MFT, INV, DIR
from checklist.expect import Expect
from checklist.pred_wrapper import PredictorWrapper
import json


def extract_data_for_test(examples, test):
    example_list = [e for e in examples if e['capability'] == test]
    data = []
    meta = []
    for e in example_list:
        data.append(e['test_case'])
        meta.append(tuple(e['target']))
    example_dict = {"data": data, "meta": meta}
    return example_list, example_dict


### added by pia ###
def predict_srl(data):
    pred = []
    for d in data:
        pred.append(srl_predictor.predict(d))
    return pred


predict_and_conf = PredictorWrapper.wrap_predict(predict_srl)


# Helper function to extract target argument
def get_arg(pred, arg_target='ARG1'):
    # we assume one predicate:
    predicate_arguments = pred['verbs'][0]
    words = pred['words']
    tags = predicate_arguments['tags']

    arg_list = []
    for t, w in zip(tags, words):
        arg = t
        if '-' in t:
            arg = t.split('-')[1]
        if arg == arg_target:
            arg_list.append(w)
    arg_set = set(arg_list)
    return arg_set


# Helper function to display failures
def format_srl(x, pred, conf, label=None, meta=None):
    try:
        return pred['verbs'][0]['description']
    except IndexError:
        return " ".join(pred['words'])


def get_arg_span(pred, target_span=[], verb_id=0):
    arg_list=[]
    if len(pred['verbs']) > verb_id:
        # we assume one predicate:
        predicate_arguments = pred['verbs'][verb_id]
        words = pred['words']
        tags = predicate_arguments['tags']

        for t, w in zip(tags, words):
            arg = t
            if '-' in t:
                arg = t.split('-')[-1]
            if w in target_span:
                arg_list.append(arg)
    return arg_list


def compare_spans_dir(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    sp_orig = meta[0].split(' ')
    sp_pred = meta[1].split(' ')
    l_orig = set(get_arg_span(orig_pred, sp_orig))
    l_pred = set(get_arg_span(pred, sp_pred))
    if l_orig == l_pred:
        pass_ = False
    else:
        pass_ = True

    return pass_


expect_fn1 = Expect.pairwise(compare_spans_dir)


def compare_spans_inv(orig_pred, pred, orig_conf, conf, labels=None, meta=None):
    sp_orig = meta[0].split(' ')
    sp_pred = meta[1].split(' ')
    l_orig = set(get_arg_span(orig_pred, sp_orig))
    l_pred = set(get_arg_span(pred, sp_pred))
    if l_orig == l_pred:
        pass_ = True
    else:
        pass_ = False

    return pass_


expect_fn4 = Expect.pairwise(compare_spans_inv)


def compare_spans_inv_passive(orig_pred, pred, orig_conf, conf, labels=None, meta=None):

    sp_orig = meta[0].split(' ')
    sp_pred = meta[1].split(' ')
    l_orig = set(get_arg_span(orig_pred, sp_orig))
    l_pred = set(get_arg_span(pred, sp_pred, verb_id=1))
    if l_orig == l_pred:
        pass_ = True
    else:
        pass_ = False

    return pass_


expect_fn5 = Expect.pairwise(compare_spans_inv_passive)


def found_arg1_object(x, pred, conf, label=None, meta=None):
    # object should be recognized as arg1
    object = meta[0]
    arg1 = get_arg(pred, arg_target='ARG1')

    if object in arg1:  # if the object is one of the 'ARG1' words
        pass_ = True
    else:
        pass_ = False
    return pass_


expect_arg3 = Expect.single(found_arg1_object)


def get_tag_from_array(pair_id, sent_id, verb_id, token_id):
    """pair_id: from data list"""
    try:
        return test.results['preds'][pair_id][sent_id]['verbs'][verb_id]['tags'][token_id].replace("I-", "").replace("B-", "")
    except IndexError:
        return ""


# Read in data
# Read in challenge set data
with open('data/srl_challenge_set.json') as f:
    data = json.load(f)
    examples = data['data']

# Extracting data for be_disambiguation
be_disam_list, be_disam_dict = extract_data_for_test(examples, "be_disambiguation")

# Extracting data for location_recognition
locrec_list, locrec_dict = extract_data_for_test(examples, "location_recognition")

# Extracting data for negating_arg1
negating_arg1_list, negating_arg1_dict = extract_data_for_test(examples, "negating_arg1")

# Extracting data for theme_in_causative_alternation
causalt_list, causalt_dict = extract_data_for_test(examples, "theme_in_causative_alternation")

# Extracting data for arg1_in_passive
passive_list, passive_dict = extract_data_for_test(examples, "arg1_in_passive")


# Performing the tests
for model in ['structured-prediction-srl', 'structured-prediction-srl-bert']:
    # allennlp srl model: https://docs.allennlp.org/models/main/models/structured_prediction/models/srl/
    # allennlp srl_bert model: https://docs.allennlp.org/models/main/models/structured_prediction/models/srl_bert/
    srl_predictor = load_predictor(model)
    model_name = "bilstm"
    if "bert" in model:
        model_name = "bert"

    output = []
    output_dict = {}  # For saving system outputs

    # Be_disambiguation
    test = DIR(**be_disam_dict, expect=expect_fn1)
    test.run(predict_and_conf)
    print(f"-----Model: {model_name}, test: be_disambiguation-----")
    test.summary(format_example_fn=format_srl)
    print()
    # Saving the predictions
    for e in be_disam_list:
        e['model'] = model_name
        # Fetching the 3rd tag for the 1st predicate in that sentence
        e['prediction'] = (get_tag_from_array(be_disam_list.index(e), 0, 0, 2),
                           get_tag_from_array(be_disam_list.index(e), 1, 0, 2))
        output.append(e)

    # Location_recognition
    test = DIR(**locrec_dict, expect=expect_fn1)
    test.run(predict_and_conf)
    print(f"-----Model: {model_name}, test: location_recognition-----")
    test.summary(format_example_fn=format_srl)
    print()
    # Saving the predictions
    for e in locrec_list:
        e['model'] = model_name
        # Fetching the 2nd-to-last tag for the 1st predicate in that sentence
        e['prediction'] = (get_tag_from_array(locrec_list.index(e), 0, 0, -2),
                           get_tag_from_array(locrec_list.index(e), 1, 0, -2))
        output.append(e)

    # Negating_arg1
    test = MFT(**negating_arg1_dict, expect=expect_arg3)
    test.run(predict_and_conf)
    print(f"-----Model: {model_name}, test: negating_arg1-----")
    test.summary(format_example_fn=format_srl)
    print()
    # Saving the predictions
    for e in negating_arg1_list:
        e['model'] = model_name
        # Fetching the 4th tag for the 1st predicate in that sentence
        e['prediction'] = test.results['preds'][negating_arg1_list.index(e)]['verbs'][0]['tags'][3].lstrip("I-")
        output.append(e)

    # theme_in_causative_alternation
    test = INV(**causalt_dict, expect=expect_fn4)
    test.run(predict_and_conf)
    print(f"-----Model: {model_name}, test: causative_alternation-----")
    test.summary(format_example_fn=format_srl)
    print()
    # Saving the predictions
    for e in causalt_list:
        e['model'] = model_name
        # Fetching the target tags
        e['prediction'] = (get_tag_from_array(causalt_list.index(e), 0, 0, 2),
                           get_tag_from_array(causalt_list.index(e), 1, 0, 0))
        output.append(e)

    # arg1_in_passive
    test = INV(**passive_dict, expect=expect_fn5)
    test.run(predict_and_conf)
    print(f"-----Model: {model_name}, test: arg1_in_passive-----")
    test.summary(format_example_fn=format_srl)
    print()
    # Saving the predictions
    for e in passive_list:
        e['model'] = model_name
        # Fetching the target tags
        e['prediction'] = (get_tag_from_array(passive_list.index(e), 0, 0, 3),
                           get_tag_from_array(passive_list.index(e), 1, 1, 1))
        output.append(e)

    # Writing output to file
    output_dict['output'] = output
    filename = f"output/output_{model_name}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
