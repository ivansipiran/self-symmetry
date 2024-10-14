import util


class Logger():
    def __init__(self, args):
        self.args = args
        pass

    def print(self, iter, iter_all):
        pass

    # log from a Loss object
    def logloss(self, data):
        pass

    def data(self):
        pass

    def write(self, filename):
        output = self.data()
        with open(self.args.save_path / filename, "w") as f:
            f.write(output)
        pass

class CDSLossLogger(Logger):
    def __init__(self, args):
        super().__init__(args)
        self.loss_history1 = []
        self.loss_history2 = []
        self.loss_history3 = []
        self.loss1_txt = ""
        self.loss2_txt = ""
        self.loss3_txt = ""

    def logloss(self, data):
        self.loss_history1.append(data[0])
        self.loss_history2.append(data[1])
        self.loss_history3.append(data[2])
        pass

    def loss_history_to_txt(self):

        str_list = []
        for i in range(len(self.loss_history1)):
            str_list.append(" ".join([
                self.convert_to_str(self.loss_history1[i]),
                self.convert_to_str(self.loss_history2[i]),
                self.convert_to_str(self.loss_history3[i]),
                                      ]))
        #self.loss1_txt = "\n".join(map(lambda x : self.convert_to_str(x), self.loss_history1))
        #self.loss2_txt = "\n".join(map(lambda x : self.convert_to_str(x), self.loss_history2))
        #self.loss3_txt = "\n".join(map(lambda x : self.convert_to_str(x), self.loss_history3))
        return "\n".join(str_list)

    def data(self):
        return self.loss_history_to_txt()

    def convert_to_str(self, val):
        return "{:.6f}".format(val)

    def print(self, iter, iter_all, print_optim = None):
        #print(f'{self.args.save_path.name}; iter: {iter} / {iter_all / self.args.batch_size}; '
        #      f'Loss1: {util.show_truncated(self.loss_history1[-1], 6)}; '
        #      f'Loss2: {util.show_truncated(self.loss_history2[-1], 6)}; '
        #      f'Loss3: {util.show_truncated(self.loss_history3[-1], 6)}')
        print_val = f'{self.args.save_path.name}; iter: {iter} / {iter_all / self.args.batch_size}; ' \
             f'Loss1: {"{:.6f}".format(self.loss_history1[-1])}; ' \
             f'Loss2: {"{:.6f}".format(self.loss_history2[-1])}; ' \
             f'Loss3: {"{:.6f}".format(self.loss_history3[-1])}'
        if print_optim is not None:
            print_val += (f'; optimizer lr : {print_optim}')

        print(print_val)



class NLogger(Logger):
    def __init__(self, args=None):
        pass

    def print(self, iter, iter_all):
        pass

    def logloss(self, string):
        pass


NullLogger = NLogger()