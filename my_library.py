def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01

def cond_probs_product(table, evidence, target_column, target_value):
  table_columns = up_list_column_names(table) 
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence)
  cond_prob_list = []
  for evidence, evidence2 in evidence_complete:
    cond_prob_list += [cond_prob(table, evidence, evidence2, target_column, target_value)]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  prob_zero = cond_probs_product(table, evidence_row, target, 0) * prior_prob(table, target, 0)
  prob_one = cond_probs_product(table, evidence_row, target, 1) * prior_prob(table, target, 1)
  neg, pos = compute_probs(prob_zero, prob_one)
  return [neg, pos]

def metrics(single_parameter):
  assert isinstance(single_parameter, list), f'Expecting Parameter to be a list but instead is {type(single_parameter)}'
  assert all(isinstance(item, list) for item in single_parameter), f'Expecting Parameter to be a list of lists'
  assert all(len(item) == 2 for item in single_parameter), f'Expecting Parameter to be a zipped list'
  assert all(isinstance(item[0], int) and isinstance(item[1], int) for item in single_parameter), f'Expecting each value in pair to be an int'
  assert all(item[0] >= 0 and item[1] >= 0 for item in single_parameter), f'Expecting each value in pair to be >= 0'
  tp = sum([1 if pair==[1,1] else 0 for pair in single_parameter])
  fp = sum([1 if pair==[1,0] else 0 for pair in single_parameter])
  fn = sum([1 if pair==[0,1] else 0 for pair in single_parameter])
  precision = 0 if (tp + fp) == 0 else tp / (tp + fp) 
  recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
  f1 = 0 if (precision + recall) == 0 else 2*(precision * recall) / (precision + recall)
  accuracy = sum([p==a for p,a in single_parameter])/len(single_parameter)
  return {'Precision': precision, 'Recall': recall, 'F1': f1, 'Accuracy': accuracy}

def try_archs(full_table, target, architectures, thresholds):
  train_table, test_table = up_train_test_split(full_table, target, .4)
  for arch in architectures:
    all_results = up_neural_net(train_table, test_table, arch, target)
  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
    pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]
  print(f'Architecture: {arch}')
  print(up_metrics_table(all_mets))
  return None
