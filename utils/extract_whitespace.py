''' Quick util for taking a "full" data file and extracting
just the data for spaces and newlines. '''
import pickle

filename = "file.pickle"

with open(filename,"rb") as f:
    data = pickle.load(f)

text = data["text"]

new_data = {"cell_states":[],"saliencies":[],"output_gate":[],"hidden_states":[],"text":text,"indices":[]}

for i, ch in enumerate(text):
     if ch not in [" ","\n"]:
         continue
     new_data["cell_states"].append(data["cell_states"][i])
     new_data["hidden_states"].append(data["hidden_states"][i])
     new_data["output_gate"].append(data["output_gate"][i])
     new_data["saliencies"].append(data["saliencies"][i])
     lensofar = len(new_data["cell_states"])
     new_data["indices"].append((i,lensofar-1))
     if i >= len(data["cell_states"]):
         break

with open("new_whitespace_data.pickle","wb") as f:
    pickle.dump(new_data,f)

