import torch
import torch.nn as nn

# class BidirectionalLSTM(nn.Module):
#
#     def __init__(self, input_size, hidden_size, output_size):
#         super(BidirectionalLSTM, self).__init__()
#         self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
#         self.linear = nn.Linear(hidden_size * 2, output_size)
#
#     def forward(self, input):
#         """
#         input : visual feature [batch_size x T x input_size]
#         output : contextual feature [batch_size x T x output_size]
#         """
#         self.rnn.flatten_parameters()
#         recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
#         output = self.linear(recurrent)  # batch_size x T x output_size
#         return output

class BidirectionalLSTMv2(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTMv2, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        # self.h0 = torch.randn(2, 1, hidden_size).cuda()
        # self.c0 = torch.randn(2, 1, hidden_size).cuda()

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        # T, b, h = recurrent.size()
        # print("recurrent.size: ", recurrent.size())
        # t_rec = recurrent.contiguous().view(T * b, h)

        output = self.linear(recurrent)  # batch_size x T x output_size
        # output = output.view(T, b, -1)
        # print("output.size: ", output.size())
        return output

class BidirectionalGRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class BidirectionalRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

# class BidirectionalGRU(nn.Module):
#
#     def __init__(self, input_size, hidden_size, output_size):
#         super(BidirectionalGRU, self).__init__()
#         self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
#         self.linear = nn.Linear(hidden_size * 2, output_size)
#
#     def forward(self, input):
#         """
#         input : visual feature [batch_size x T x input_size]
#         output : contextual feature [batch_size x T x output_size]
#         """
#         self.rnn.flatten_parameters()
#         recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
#         output = self.linear(recurrent)  # batch_size x T x output_size
#         return output