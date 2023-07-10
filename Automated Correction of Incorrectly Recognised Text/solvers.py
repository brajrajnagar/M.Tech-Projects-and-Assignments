import queue
class SentenceCorrector(object):
    def __init__(self, cost_fn, conf_matrix):
        self.conf_matrix = conf_matrix
        self.cost_fn = cost_fn

        # You should keep updating following variable with best string so far.
        self.best_state = None  




    def search(self, state):
        def getCost(words,i): 
            states=getStates(words,i)
            cost=0
            for st in states:
                cost+=self.cost_fn(st)
            return cost

        def getStates(words,i):
            states=[]
            n=len(words)
            for j in range(max(i-2,0),n):
                if(j+2>n):
                    break
                states.append(" ".join(words[j:j+2]))
            return states
        def update(words):
            state = " ".join(words)
            oc = self.cost_fn(self.best_state)
            nc = self.cost_fn(state)
            if nc < oc :
                self.best_state=state
                return True
            return False

        dict_rev = {}
        for i in range(97,97+26):
            dict_rev[chr(i)] = []

        for corr in self.conf_matrix.keys():
            for rong in self.conf_matrix[corr]:
                dict_rev[rong].append(corr)
        self.conf_matrix = dict_rev.copy()

        self.best_state=state
        words=state.split()
        n = len(words)
        for i in range(n):
            word=words[i]
            m=len(word)
            old_cost=getCost(words,i)
            for j in range(0,m):
                for ch in self.conf_matrix[word[j]]:
                    words[i]=word[:j]+ch+word[j+1:]
                    new_cost=getCost(words,i)
                    if new_cost < old_cost and update(words):
                        old_cost=new_cost
                        word=words[i]
            for j in range(0,m):
                for ch in self.conf_matrix[word[j]]:
                    for k in range(j+1,m):
                        for ch2 in self.conf_matrix[word[k]]:
                            words[i]=word[:j]+ch+word[j+1:k]+ ch2 + word[k+1:]
                            new_cost=getCost(words,i)
                            if new_cost < old_cost and update(words):
                                old_cost=new_cost
                                word=words[i]
            words[i]=word




