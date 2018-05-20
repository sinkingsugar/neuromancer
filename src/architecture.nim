import random, options, sequtils, math, algorithm
import methods

type
  Connection* = ref object
    nodeFrom: Node
    nodeTo: Node
    gain: float
    
    weight: float

    gater: Node
    elegibility: float

    previousDeltaWeight: float
    totalDeltaWeight: float

    xtrace: tuple[
      nodes: seq[Node], 
      values: seq[float]
      ]

  NodeKind* {.pure.} = enum
    Input, Output, Hidden, Constant

  Network* = object
    nodes: seq[Layer]
    input: Layer
    output: Layer

  Layer* = ref object
    nodes: seq[Node]
    
    connections: tuple[
      inbound: seq[Connection],
      outbound: seq[Connection],
      self: seq[Connection]
    ]
  
  Node* = ref object
    bias: float
    squash: Activation
    kind: NodeKind
    
    activation: float
    derivative: float
    state: float
    old: float

    mask: float

    previousDeltaBias: float
    totalDeltaBias: float

    connections: tuple[
      inbound: seq[Connection],
      outbound: seq[Connection],
      gated: seq[Connection],
      self: Connection
    ]

    error: tuple[
      responsibility: float,
      projected: float,
      gated: float
    ]

proc newConnection*(nodeFrom, nodeTo: Node; weight: float): Connection =
  new result
  result.nodeFrom = nodeFrom
  result.nodeTo = nodeTo
  result.gain = 1

  result.weight = weight

  result.gater = nil

  result.xtrace.nodes = newSeq[Node]()
  result.xtrace.values = newSeq[float]()

proc newConnection*(nodeFrom, nodeTo: Node): Connection = newConnection(nodeFrom, nodeTo, rand(1.0) * 0.2 - 0.1)

proc newNode*(kind: NodeKind = NodeKind.Hidden): Node =
  new result

  result.bias = if kind == NodeKind.Input: 0.0 else: rand(1.0) * 0.2 - 0.1
  result.squash = Activation.logistic
  result.kind = kind

  result.mask = 1
  
  result.connections.inbound = newSeq[Connection]()
  result.connections.outbound = newSeq[Connection]()
  result.connections.gated = newSeq[Connection]()
  result.connections.self = newConnection(result, result, 0)

proc newLayer*(size: int; kind: NodeKind = NodeKind.Hidden): Layer =
  new result

  result.nodes = newSeq[Node]()

  result.connections.inbound = newSeq[Connection]()
  result.connections.outbound = newSeq[Connection]()
  result.connections.self = newSeq[Connection]()

  for i in 0..<size:
    result.nodes.insert(newNode(kind))

func innovationID*(connection: Connection; a, b: float): float =
  # Returns an innovation ID
  # https://en.wikipedia.org/wiki/Pairing_function (Cantor pairing function)
  1 / 2 * (a + b) * (a + b + 1) + b

proc activate*(self: Node): float =
  #  if (training) this.mask = Math.random() < this.dropout ? 0 : 1;

  self.old = self.state

  # All activation sources coming from the self itself
  self.state = self.connections.self.gain * self.connections.self.weight * self.state + self.bias

  # Activation sources coming from connections
  for connection in self.connections.inbound:
    self.state += connection.nodeFrom.activation * connection.weight * connection.gain

  self.activation = self.squash.forward(self.state) * self.mask
  self.derivative = self.squash.derivative(self.state)

  var
    nodes = newSeq[Node]()
    influences = newSeq[float]()
  
  for connection in self.connections.gated:
    let
      node = connection.nodeTo
      index = nodes.find(node)
    
    if index > -1:
      influences[index] += connection.weight * connection.nodeFrom.activation
    else:
      nodes.insert(node)
      let plus = if node.connections.self.gater == self: node.old else: 0
      influences.insert(connection.weight * connection.nodeFrom.activation + plus)
    
    connection.gain = self.activation
  
  for connection in self.connections.inbound:
    connection.elegibility = 
      self.connections.self.gain * 
      self.connections.self.weight * 
      connection.elegibility + connection.nodeFrom.activation * 
      connection.gain
    
    for i in 0..<nodes.len:
      let
        node = nodes[i]
        influence = influences[i]
        index = connection.xtrace.nodes.find(node)
      
      if index > -1:
        connection.xtrace.values[index] =
          node.connections.self.gain *
          node.connections.self.weight *
          connection.xtrace.values[index] + self.derivative *
          connection.elegibility *
          influence
      else:
        connection.xtrace.nodes.insert(node)
        connection.xtrace.values.insert(self.derivative * connection.elegibility * influence)
  
  return self.activation

proc activate*(self: Node; input: float): float =
  self.activation = input
  return input

proc activate*(self: Layer): seq[float] =
  result = newSeq[float]()
  
  for node in self.nodes:
    result.insert(node.activate())

proc activate*(self: Layer; input: openarray[float]): seq[float] =
  assert(input.len == self.nodes.len)

  result = newSeq[float]()
  
  for i in 0..<self.nodes.len:
    let node = self.nodes[i]
    result.insert(node.activate(input[i]))

proc activate*(self: Network; input: openarray[float]): seq[float] =
  discard self.nodes[0].activate(input)
  for i in 1..<self.nodes.len - 1:
    discard self.nodes[i].activate()
  return self.nodes[^1].activate()

proc propagate*(self: Node; rateOption, momentumOption: Option[float]; update: bool; target: float = 0.0) =
  let
    momentum = if momentumOption.isSome: momentumOption.get else: 0.0
    rate = if rateOption.isSome: rateOption.get else: 0.3
  
  var error = 0.0

  if self.kind == NodeKind.Output: # Output nodes get their error from the enviroment
    self.error.responsibility = target - self.activation
    self.error.projected = self.error.responsibility
  else:
    for connection in self.connections.outbound:
      let node = connection.nodeTo
      error += node.error.responsibility * connection.weight * connection.gain
  
    # Projected error responsibility
    self.error.projected = self.derivative * error

    # Error responsibilities from all connections gated by self neuron
    error = 0.0

    for connection in self.connections.gated:
      let node = connection.nodeTo
      
      var influence = if node.connections.self.gater == self: node.old else: 0
      
      influence += connection.weight * connection.nodeFrom.activation
      error += node.error.responsibility * influence

    self.error.gated = self.derivative * error

    self.error.responsibility = self.error.projected + self.error.gated
  
  if self.kind == NodeKind.Constant: return

  for connection in self.connections.inbound:
    var gradient = self.error.projected * connection.elegibility

    for i in 0..<connection.xtrace.nodes.len:
      let
        node = connection.xtrace.nodes[i]
        value = connection.xtrace.values[i]
      
      gradient += node.error.responsibility * value

    let deltaWeight = rate * gradient * self.mask
    connection.totalDeltaWeight = deltaWeight
    if update:
      connection.totalDeltaWeight += momentum * connection.previousDeltaWeight
      connection.weight += connection.totalDeltaWeight
      connection.previousDeltaWeight = connection.totalDeltaWeight
      connection.totalDeltaWeight = 0
  
  let deltaBias = rate * self.error.responsibility
  self.totalDeltaBias += deltaBias
  if update:
    self.totalDeltaBias += momentum * self.previousDeltaBias
    self.bias += self.totalDeltaBias
    self.previousDeltaBias = self.totalDeltaBias
    self.totalDeltaBias = 0

proc propagate*(self: Layer; rateOption, momentumOption: Option[float]; update: bool; targets: openarray[float]) =  
  assert(self.nodes.len == targets.len)

  for i in 0..<self.nodes.len:
    let node = self.nodes[i]
    node.propagate(rateOption, momentumOption, true, targets[i])

proc propagate*(self: Layer; rateOption, momentumOption: Option[float]; update: bool) =  
  for i in 0..<self.nodes.len:
    let node = self.nodes[i]
    node.propagate(rateOption, momentumOption, true)

proc propagate*(self: Network; rateOption, momentumOption: Option[float]; update: bool; targets: openarray[float]) =
  self.nodes[^1].propagate(rateOption, momentumOption, update, targets)
  for i in countdown(self.nodes.len - 2, 0):
    self.nodes[i].propagate(rateOption, momentumOption, update)

proc isProjectingTo(self, target: Node): bool =
  if target == self and self.connections.self.weight != 0:
    return true
  
  for connection in self.connections.outbound:
    if connection.nodeTo == target:
      return true
  
  return false

proc isProjectedBy*(self, node: Node): bool =
  if node == self and self.connections.self.weight != 0:
    return true
  
  for connection in self.connections.inbound:
    if connection.nodeFrom == node:
      return true
  
  return false

proc gate*(self: Node; connections: Connection | seq[Connection]) =
  when connections is Connection:
    self.connections.gated.insert(connections)
    connections.gater = self
  elif connections is seq[Connection]:
    for connection in connections:
      self.gate(connections)

proc gate*(self: Layer; connections: Connection | seq[Connection]; kind: GatingKind) = 
  var 
    conns: seq[Connection]
    nodes1 = newSeq[Node]()
    nodes2 = newSeq[Node]()

  when connections is Connection:
    conns = @[connections]
  else:
    conns = connections
  
  for connection in conns:
    if not nodes1.contains(connection.nodeFrom):
      nodes1.insert(connection.nodeFrom)
    if not nodes2.contains(connection.nodeTo):
      nodes2.insert(connection.nodeTo)
  
  case kind
  of GatingKind.Input:
    for i in 0..<nodes2.len:
      let
        node = nodes2[i]
        gater = self.nodes[i mod self.nodes.len]
      
      for connection in node.connections.inbound:
        if connections.contains(connection):
          gater.gate(connection)
  of GatingKind.Output:
    for i in 0..<nodes1.len:
      let
        node = nodes1[i]
        gater = self.nodes[i mod self.nodes.len]
      
      for connection in node.connections.outbound:
        if connections.contains(connection):
          gater.gate(connection)
  of GatingKind.Self:
    for i in 0..<nodes1.len:
      let
        node = nodes1[i]
        gater = self.nodes[i mod self.nodes.len]
      
      if connections.contains(node.connections.self):
        gater.gate(node.connections.self)

proc ungate*(self: Node; connections: Connection | seq[Connection]) =
  when connections is Connection:
    let index = self.connections.gated.find(connections)
    self.connections.gated.delete(index)
    connections.gater = nil
    connections.gain = 1
  elif connections is seq[Connection]:
    for connection in connections:
      self.ungate(connections)

proc connect*(self, target: Node | Layer; kindOption: Option[ConnectionKind] = ConnectionKind.none; weight: float = 1): seq[Connection] =
  result = newSeq[Connection]()

  when self is Layer and target is Layer:
    var kind = ConnectionKind.AllToAll

    if kindOption.isNone:
      if self == target:
        kind = ConnectionKind.OneToOne
    else:
      kind = kindOption.get

    if kind == ConnectionKind.AllToAll or kind == ConnectionKind.AllToElse:
      for i in 0..<self.nodes.len:
        for j in 0..<target.nodes.len:
          if kind == ConnectionKind.AllToElse and self.nodes[i] == target.nodes[j]:
            continue
          let connection = self.nodes[i].connect(target.nodes[j], kindOption, weight)
          self.connections.outbound.insert(connection)
          target.connections.inbound.insert(connection)
          result.insert(connection)
    elif kind == ConnectionKind.OneToOne:
      assert(self.nodes.len == target.nodes.len)

      for i in 0..<self.nodes.len:
        let connection = self.nodes[i].connect(target.nodes[i], kindOption, weight)
        self.connections.self.insert(connection)
        result.insert(connection)
  elif self is Layer and target is Node:
    for node in self.nodes:
      let connection = node.connect(target, weight)
      self.connections.outbound.insert(connection)
      result.insert(connection)
  elif self is Node and target is Layer:
    for subNode in target.nodes:
      let connection = newConnection(self, subNode, weight)
      subNode.connections.inbound.insert(connection)
      self.connections.outbound.insert(connection)
      target.connections.inbound.insert(connection)
      result.insert(connection)
  elif self is Node and target is Node:
    if self == target:
      if self.connections.self.weight != 0:
        echo "This connection already exists!"
      else:
        self.connections.self.weight = weight
  
      result.insert(self.connections.self)
    elif self.isProjectingTo(target):
      raiseAssert("Already projecting a connection to self node!")
    else:
      let connection = newConnection(self, target, weight)
      target.connections.inbound.insert(connection)
      self.connections.outbound.insert(connection)
      
      result.insert(connection)

proc disconnect*(self, target: Node | Layer; twosided: bool = false) =
  when self is Node and target is Node:
    if self == target:
      self.connections.self.weight = 0
      return
    
    for i in countdown(self.connections.outbound.len - 1, 0):
      let connection = self.connections.outbound[i]
      if connection.nodeTo == target:
        self.connections.outbound.delete(i)
        let j = connection.nodeTo.connections.inbound.find(connection)
        connection.nodeTo.connections.inbound.delete(j)
        if connection.gater != nil:
          connection.gater.ungate(connection)
        break
    
    if twosided:
      target.disconnect(self)
  elif self is Layer and target is Layer:
    for i in 0..<self.nodes.len:
      for j in 0..<target.nodes.len:
        self.nodes[i].disconnect(target.nodes[j], twosided)

        for k in countdown(self.connections.outbound.len - 1, 0):
          let connection = self.connections.outbound[k]

          if connection.nodeFrom == self.nodes[i] and connection.nodeTo == target.nodes[j]:
            self.connections.outbound.delete(k)
            break
        
        if twosided:
          for k in countdown(self.connections.inbound.len - 1, 0):
            let connection = self.connections.inbound[k]

            if connection.nodeFrom == target.nodes[j] and connection.nodeTo == self.nodes[i]:
              self.connections.inbound.delete(k)
              break
  elif self is Layer and target is Node:
    for node in self.nodes:
      node.disconnect(target, twosided)
    
      for k in countdown(self.connections.outbound.len - 1, 0):
        let connection = self.connections.outbound[k]

        if connection.nodeFrom == node and connection.nodeTo == target:
          self.connections.outbound.delete(k)
          break
      
      if twosided:
        for k in countdown(self.connections.inbound.len - 1, 0):
          let connection = self.connections.inbound[k]

          if connection.nodeFrom == target and connection.nodeTo == node:
            self.connections.inbound.delete(k)
            break

proc clear*(self: Node | Layer | Network) =
  when self is Node:
    for connection in self.connections.inbound:
      connection.elegibility = 0
      connection.xtrace.nodes = @[]
      connection.xtrace.values = @[]

    for connection in self.connections.gated:
      connection.gain = 0
    
    self.error.responsibility = 0
    self.error.projected = 0
    self.error.gated = 0

    self.old = 0
    self.state = 0
    self.activation = 0
  elif self is Layer:
    for node in self.nodes:
      node.clear()
  elif self is Network:
    for layer in self.nodes:
      layer.clear()

proc mutate*(self: Node; kind: MutationKind) =
  case kind
  of MutationKind.Activation:
    let 
      current = ActivationMutation.allowed.find(self.squash)
      newSquash = (current.float + floor(rand(1.0) * (ActivationMutation.allowed.len - 1).float) + 1).int mod ActivationMutation.allowed.len
    self.squash = ActivationMutation.allowed[newSquash]
  of MutationKind.Bias:
    self.bias = rand(1.0) * (BiasMutation.max - BiasMutation.min) + BiasMutation.min
  else:
    discard

proc Memory*(_: typedesc[Layer]; size, memory: int): seq[Layer] =
  var previous: Layer = nil
  result = newSeq[Layer]()
  for i in 0..<memory:
    let layer = newLayer(size)

    for node in layer.nodes:
      node.squash = Activation.identity
      node.bias = 0
      node.kind = NodeKind.Constant
    
    if previous != nil:
      discard previous.connect(layer, ConnectionKind.OneToOne.some, 1.0)
    
    previous = layer
    result.insert(layer)
  
  result.reverse()

  for layer in result:
    layer.nodes.reverse()

proc Perceptron*(_: typedesc[Network]; inputSize: int; hiddenSize: openarray[int]; outputSize: int): Network =
  let
    input = newLayer(inputSize, NodeKind.Input)
    output = newLayer(outputSize, NodeKind.Output)

  var 
    network = newSeq[Layer]()
    previous = input
  
  network &= input

  for i in 0..<hiddenSize.len:
    let layer = newLayer(hiddenSize[i], NodeKind.Hidden)

    network &= layer
    discard previous.connect(layer, ConnectionKind.AllToAll.some)
    
    previous = layer
  
  network &= output
  discard previous.connect(output, ConnectionKind.AllToAll.some)

  result.nodes = network
  result.input = input
  result.output = output

proc NARX*(_: typedesc[Network]; inputSize: int; hiddenSize: openarray[int]; outputSize, inputMemory, outputMemory: int): Network =
  let
    input = newLayer(inputSize, NodeKind.Input)
    inputMemory = Layer.Memory(inputSize, inputMemory)
    output = newLayer(outputSize, NodeKind.Output)
    outputMemory = Layer.Memory(outputSize, outputMemory)
  
  var 
    network = newSeq[Layer]()
    previous = input
    firstHidden: Layer
  
  network &= input
  network &= outputMemory

  for i in 0..<hiddenSize.len:
    let layer = newLayer(hiddenSize[i], NodeKind.Hidden)

    if firstHidden == nil:
      firstHidden = layer

    network &= layer
    discard previous.connect(layer, ConnectionKind.AllToAll.some)
    
    previous = layer

  network &= inputMemory
  network &= output
  discard previous.connect(output, ConnectionKind.AllToAll.some)
  
  discard input.connect(inputMemory[^1], ConnectionKind.OneToOne.some, 1.0)
  discard inputMemory[^1].connect(firstHidden)
  discard output.connect(outputMemory[^1], ConnectionKind.OneToOne.some, 1.0)
  discard outputMemory[^1].connect(firstHidden)

  result.nodes = network
  result.input = input
  result.output = output

proc LSTM*(_: typedesc[Network]; inputSize: int; lstmSize: openarray[int]; outputSize: int): Network =
  let
    input = newLayer(inputSize, NodeKind.Input)
    output = newLayer(outputSize, NodeKind.Output)

  var 
    network = newSeq[Layer]()
    previous = input

  network &= input

  for i in 0..<lstmSize.len:
    let
      inputGate = newLayer(lstmSize[i])
      forgetGate = newLayer(lstmSize[i])
      memoryCell = newLayer(lstmSize[i])
      outputGate = newLayer(lstmSize[i])
      outputBlock = if i == lstmSize.len - 1: output else: newLayer(lstmSize[i])
  
    for node in inputGate.nodes:
      node.bias = 1

    for node in forgetGate.nodes:
      node.bias = 1
    
    for node in outputGate.nodes:
      node.bias = 1

    let inputConn = previous.connect(memoryCell, ConnectionKind.AllToAll.some)
    discard previous.connect(inputGate, ConnectionKind.AllToAll.some)
    discard previous.connect(outputGate, ConnectionKind.AllToAll.some)
    discard previous.connect(forgetGate, ConnectionKind.AllToAll.some)

    discard memoryCell.connect(inputGate, ConnectionKind.AllToAll.some)
    discard memoryCell.connect(forgetGate, ConnectionKind.AllToAll.some)
    discard memoryCell.connect(outputGate, ConnectionKind.AllToAll.some)

    let forgetConn = memoryCell.connect(memoryCell, ConnectionKind.OneToOne.some)
    let outputConn = memoryCell.connect(outputBlock, ConnectionKind.AllToAll.some)
    
    inputGate.gate(inputConn, GatingKind.Input)
    forgetGate.gate(forgetConn, GatingKind.Self)
    outputGate.gate(outputConn, GatingKind.Output)

    network &= inputGate
    network &= forgetGate
    network &= memoryCell
    network &= outputGate
    network &= outputBlock

    previous = outputBlock

  result.nodes = network
  result.input = input
  result.output = output

when isMainModule:
  when defined(NARX):
    let
      network = Network.NARX(2, [4], 1, 4, 4)

    for i in 0..5_00:
      echo network.activate([0.0, 0.0]), " ", 1.0
      network.propagate(float.none, float.none, true, [1.0])

      echo network.activate([0.0, 1.0]), " ", 0.0
      network.propagate(float.none, float.none, true, [0.0])

      echo network.activate([1.0, 0.0]), " ", 0.0
      network.propagate(float.none, float.none, true, [0.0])

      echo network.activate([1.0, 1.0]), " ", 1.0
      network.propagate(float.none, float.none, true, [1.0])
  elif defined(LSTM):
    let
      network = Network.LSTM(1, [6, 3], 1)

    for i in 0..6_000:
      echo network.activate([0.0]), " ", 0.0
      network.propagate(float.none, float.none, true, [0.0])

      echo network.activate([1.0]), " ", 1.0
      network.propagate(float.none, float.none, true, [1.0])

      echo network.activate([1.0]), " ", 0.0
      network.propagate(float.none, float.none, true, [0.0])

      echo network.activate([0.0]), " ", 1.0
      network.propagate(float.none, float.none, true, [1.0])

      echo network.activate([0.0]), " ", 0.0
      network.propagate(float.none, float.none, true, [0.0])

      network.clear()
  else:
    let
      network = Network.Perceptron(2, [4, 3], 1)

    for i in 0..50_000:
      echo network.activate([0.0, 0.0]), " ", 1.0
      network.propagate(float.none, float.none, true, [1.0])

      echo network.activate([0.0, 1.0]), " ", 0.0
      network.propagate(float.none, float.none, true, [0.0])

      echo network.activate([1.0, 0.0]), " ", 0.0
      network.propagate(float.none, float.none, true, [0.0])

      echo network.activate([1.0, 1.0]), " ", 1.0
      network.propagate(float.none, float.none, true, [1.0])
