# GraphRNN
Codes for the paper "Learning Graph-Level Representations with Gated Recurrent Neural Networks". (https://arxiv.org/pdf/1805.07683.pdf)

# 1. Setup

Get the source code,

    git clone https://github.com/yuj-umd/graphRNN.git

Install pytorch from https://pytorch.org/

# 2. Usage

Run

    python main.py \
            -seed 1 \
		-data $data \
		-learning_rate $learning_rate \
		-num_epochs $num_epochs \
		-hidden $hidden \
		-fold $fold \
		-embedding_dim $embedding_dim \
		-rnn_hidden_dim $rnn_hidden_dim 

Paramaters are defined as

    data: MUTAG, NCI1, NCI109, DD, ENZYMES
    feat_dim: Number of node labels
    embedding_dim: Dimension of node embedding
    num_class: Number of graph classes
    rnn_hidden_dim: Hidden unit size of RNN
    learning_rate: initial learning_rate



# 3. Reference
    @article{jin2018learning,
        title={Learning Graph-Level Representations with Gated Recurrent Neural Networks},
        author={Jin, Yu and JaJa, Joseph F},
        journal={arXiv preprint arXiv:1805.07683},
        year={2018}
     }
