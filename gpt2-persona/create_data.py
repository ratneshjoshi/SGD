import pandas as pd

df = pd.read_csv("data/personality.csv", index_col=False)

print(df.columns)

num_conversations = len(df)

num_train = .9*num_conversations

data_train = []
data_val = []

for i, row in df.iterrows():

	persona = row['Persona']
	utterances = row['chat'].split('\n')
	num_utterances = len(utterances)

	# get indeces from conversation for training/validation data
	train_indices = [(i,i+1,i+2,i+3) for i in range(0, num_utterances, 2) if i+3<num_utterances]
	# print(train_indices)

	# all data from 1 conversation
	data = []
	for indices in train_indices:
		c1, c2, c3, r = indices
		data_point = '[PERSONA] ' + persona + '[CONTEXT] [speaker1] '+ utterances[c1] +  \
						' [speaker2] ' + utterances[c2] + ' [speaker1] ' + utterances[c3] + ' [END_OF_CONTEXT] ' + \
						' [RESPONSE] ' + ' [speaker2] ' + utterances[r] + ' [END_OF_RESPONSE] '
		
		data.append(data_point)

	if i > num_train:
		data_val = data_val + data
	else:
		data_train = data_train + data

	if (i+1)%100==0:
		print(f'{i+1} out of {num_conversations} done')



train_df = pd.DataFrame.from_dict({'text':data_train})
val_df = pd.DataFrame.from_dict({'text':data_val})

train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)


