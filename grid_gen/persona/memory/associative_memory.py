
class AssociativeMemory: 
  def __init__(self, f_saved): 
    self.id_to_node = dict()

    self.seq_event = []
    self.seq_thought = []
    self.seq_chat = []

    self.kw_to_event = dict()
    self.kw_to_thought = dict()
    self.kw_to_chat = dict()

    self.kw_strength_event = dict()
    self.kw_strength_thought = dict()


