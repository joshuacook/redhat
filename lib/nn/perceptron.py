from numpy import asarray, insert, ndarray, piecewise
class Perceptron(ndarray):
    
    def __new__(cls, weights):
        return asarray(weights).view(cls)
    
    def activate(self,inputs):
        try:
            assert(len(inputs)+1==len(self))
        except:
            return 'input vector of dimension {} is not valid for a bias/weight vector of dimension {}'.format(
                len(inputs), len(self))
        inputs = insert(inputs,0,-1)
        return self._threshold(self.dot(inputs))

    def update(self,inputs,target,eta=.1):

        result = self.activate(inputs)
        inputs = insert(inputs,0,-1)
        self[1:] += eta*(target-result)*inputs[1:]
          
    def _threshold(self, energy):
            
        return piecewise(energy, [energy < 0, energy >= 0], [0 , 1])  
                           
            
