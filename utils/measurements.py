''' Extract information about the model's hidden states when modeling text.
Also includes functions for extracting some data (e.g. integrated gradients)
that has nothing to do with the final paper I presented.  Sorry.'''

import unidecode
import time

from charlstm import *

def tweak_whitespace(s):
    ''' Replace whitespace characters with visible characters. '''
    return s.replace("\n","\\n").replace(" ","_")

def get_stats(model, input_sequence, layer=0,verbose=False,
        use_loss=True,fast_mode=False,minimal=False):
    ''' For a given input sequence, return data representing the 
    hidden state, cell state, cell salience, and 'integrated gradients' 
    of loss function (or log probability of predicted class)
    (wrt previous cell state) at each time step.
    Input_sequence should be a string.

    if use_loss is True, saliency and integrated gradients are taken for
    the negative log loss function.  If false, we instead measure with respect
    to the log probability of the predicted class.
    
    You must specify which layer of the LSTM to return data for, 
    though all will be computed anyways.  Sorry.'''
    cell_states = []
    outputs = []
    hidden_states = []
    saliencies = []
    integrated_gradients = []
    predictions = []
    output_gate = []

    loss_function = nn.NLLLoss()

    h, cell = model.init_hidden()

    for i in range(len(input_sequence)-1):
        start_time = time.time()
        seq = model.text_to_sequence(input_sequence[i])
        target = model.text_to_sequence(input_sequence[i+1]).view(-1)

        # Get basic data, incl. the hidden state for next iteration
        output, (new_h, new_cell) = model(seq, (h, cell))

        cell_states.append(new_cell[layer,0,:].cpu().detach().numpy())
        hidden_states.append(new_h[layer,0,:].cpu().detach().numpy())
        outputs.append(output[-1].cpu().detach().numpy())

        # Calculate output gate values. 
        #  Note: h_t = tanh(c_t) * o_t, so o_t = h_t/tanh(c_t)
        c_curr = cell_states[-1]
        h_curr = hidden_states[-1]
        output_gate.append(h_curr/np.tanh(c_curr))

        # Record a text summary of some possible predictions by the model
        #  at this point in time.
        if not fast_mode:
            topv, topi = output[-1].topk(4)
            topv = topv.exp()
            chars = [model.charset[j] for j in topi]
            snippets = ""
            for j, char in enumerate(chars):
                # Generate the possible continuations based on each character
                snippet = model.gen_text(char,hidden=(new_h,new_cell),n_steps=8)
                snippet = unidecode.unidecode(tweak_whitespace(snippet)) 
                snippets = snippets + "%s: %.3f\n" % (snippet,topv[j])
            predictions.append(snippets)

        # Get saliency (absolute value of gradient)
        var_cell = torch.autograd.Variable(cell, requires_grad=True)
        output, (temp_h, temp_c) = model(seq, (h, var_cell)) 
        loss = loss_function(output, target) if use_loss else get_score(output)
        grads = torch.autograd.grad(loss, var_cell,retain_graph=False)
        saliency = grads[0].abs()
        saliencies.append(saliency[layer,0,:].cpu().detach().numpy())

        if not fast_mode:
            # Get integrated gradients wrt previous cell state.
            #  (as in https://arxiv.org/abs/1703.01365)
            model.zero_grad()
            baseline_cell = torch.autograd.Variable(torch.zeros(cell.size()), requires_grad=True)
            var_cell = torch.autograd.Variable(cell, requires_grad=True)

            m = 100 # Number of steps/samples to use
            cumulative = torch.zeros(cell.size()) # i.e. the zero vector
            for k in range(1, m+1):
                point = baseline_cell + (k/m)*(var_cell - baseline_cell)
                output, (temp_h, temp_c) = model(seq, (h, point))
                loss = loss_function(output, target) if use_loss else get_score(output)
                grads = torch.autograd.grad(loss, point, retain_graph=False)
                cumulative += grads[0]
            final = (cell - baseline_cell)*cumulative*(1/m)
            integrated_gradients.append(final[layer,0,:].cpu().detach().numpy())


        # Update for next iteration
        h = new_h.detach()
        cell = new_cell.detach()

        if verbose:
            print("Finished getting stats for item %d" % i)
            print("Time elapsed: %.2fs" % (time.time() - start_time) )

    cell_states = torch.tensor(cell_states)
    hidden_states = torch.tensor(hidden_states)
    outputs = torch.tensor(outputs)
    saliencies = torch.tensor(saliencies)
    integrated_gradients = torch.tensor(integrated_gradients)
    output_gate = torch.tensor(output_gate)

    if fast_mode:
        return cell_states, hidden_states, outputs, saliencies, output_gate
    else:
        return cell_states, hidden_states, saliencies, integrated_gradients, predictions, output_gate

def get_whitespace_stats(model, input_sequence, layer=0, verbose=False):
    ''' Like get_stats, but less in-depth.  We only save stats for
        whitespace characters, and fewer values are computed.
        
        Return a list of data along with a list of indices describing which
        points in the text these data points correspond to.'''
    data = []
    hidden_states = []
    cell_states = []
    indices = []
    h, cell = model.init_hidden()

    for i in range(len(input_sequence)-1):
        start_time = time.time()
        seq = model.text_to_sequence(input_sequence[i])
        is_space = input_sequence[i] in [" ","\n"]

        # Get basic data, incl. the hidden state for next iteration
        output, (new_h, new_cell) = model(seq, (h, cell))

        if is_space:
            # Calculate output gate values. 
            #  Note: h_t = tanh(c_t) * o_t, so o_t = h_t/tanh(c_t)
            c_curr = new_cell[layer,0,:].cpu().detach().numpy()
            h_curr = new_h[layer,0,:].cpu().detach().numpy()
            output_gate_value = h_curr/np.tanh(c_curr)
            data.append(output_gate_value)
            hidden_states.append(h_curr)
            cell_states.append(c_curr)
            indices.append(i)

        # Update for next iteration
        h = new_h.detach()
        cell = new_cell.detach()

        if verbose:
            print("Finished getting stats for item %d" % i)
            print("Time elapsed: %.2fs" % (time.time() - start_time) )

    data = torch.tensor(data)
    hidden_states = torch.tensor(hidden_states)
    cell_states = torch.tensor(cell_states)

    return data, indices, hidden_states, cell_states

def get_stats_minimal(model, input_sequence, layer=0,verbose=False):
    ''' For a given sequence of input tokens, return just the output gate
    values.
    
    Probably should not actually be used.'''
    output_gate = []
    h, cell = model.init_hidden()
    with torch.no_grad():
        for i in range(len(input_sequence)-1):
            start_time = time.time()
            seq = model.text_to_sequence(input_sequence[i])

            output, (h, cell) = model(seq, (h, cell))
            c_curr = cell[layer,0,:].cpu().numpy()
            h_curr = h[layer,0,:].cpu().numpy()

            # Calculate output gate values. 
            #  Note: h_t = tanh(c_t) * o_t, so o_t = h_t/tanh(c_t)
            output_gate.append(h_curr/np.tanh(c_curr))

            if verbose:
                print("Finished getting stats for item %d" % i)
                print("Time elapsed: %.2fs" % (time.time() - start_time) )
    return np.array(output_gate)


def get_score(output):
    ''' Return the probability of the predicted class. '''
    return torch.exp(output.max())

def get_perplexity(text, model, avg_word_length):
    ''' Get cross entropy, bits-per-character, and an 
    equivalent per-word perplexity. '''
    # Try to use CUDA if available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)
    model.set_device(device)

    model.eval() # Make sure model is in evaluation mode
    h, c = model.init_hidden()

    # Run the model 
    input_sequence = model.text_to_sequence(text[:-1]).to(device)
    target = model.text_to_sequence(text[1:]).view(-1).to(device)
    output, _ = model(input_sequence,(h.to(device),c.to(device)))

    negative_log_loss = nn.NLLLoss()
    cross_entropy = float(negative_log_loss(output, target).cpu())

    # Get BPC (switch from natural log to base 2 log)
    #  average(log_2(p)) = average(ln(p)/ln(2)) = average(ln(p))/ln(2)
    bits_per_character = cross_entropy/np.log(2)

    # Calculate equivalent perplexity-per-word
    ppl = np.exp(avg_word_length*cross_entropy)
    # Could also be calculated in a different but equivalent way:
    #ppl = 2**(avg_word_length*bits_per_character)

    return cross_entropy, bits_per_character, ppl
