import random, options, sequtils, math
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

  NodeGroup* = ref object
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

proc newGroup*(size: int): NodeGroup =
  new result

  result.nodes = newSeq[Node]()

  result.connections.inbound = newSeq[Connection]()
  result.connections.outbound = newSeq[Connection]()
  result.connections.self = newSeq[Connection]()

  for i in 0..<size:
    result.nodes.insert(newNode())

# Returns an innovation ID
# https://en.wikipedia.org/wiki/Pairing_function (Cantor pairing function)
func innovationID*(connection: Connection; a, b: float): float = 1 / 2 * (a + b) * (a + b + 1) + b

proc activate*(this: Node): float =
  this.old = this.state

  # All activation sources coming from the this itself
  this.state = this.connections.self.gain * this.connections.self.weight * this.state + this.bias

  # Activation sources coming from connections
  for connection in this.connections.inbound:
    this.state += connection.nodeFrom.activation * connection.weight * connection.gain

  this.activation = this.squash.forward(this.state) * this.mask
  this.derivative = this.squash.derivative(this.state)

  var
    nodes = newSeq[Node]()
    influences = newSeq[float]()
  
  for connection in this.connections.gated:
    let
      this = connection.nodeTo
      index = nodes.find(this)
    
    if index > -1:
      influences[index] += connection.weight * connection.nodeFrom.activation
    else:
      nodes.insert(this)
      let plus = if this.connections.self.gater == this: this.old else: 0
      influences.insert(connection.weight * connection.nodeFrom.activation + plus)
    
    connection.gain = this.activation
  
  for connection in this.connections.inbound:
    connection.elegibility = 
      this.connections.self.gain * 
      this.connections.self.weight * 
      connection.elegibility + connection.nodeFrom.activation * 
      connection.gain
    
    for i in 0..<nodes.len:
      let
        this = nodes[i]
        influence = influences[i]
        index = connection.xtrace.nodes.find(this)
      
      if index > -1:
        connection.xtrace.values[index] =
          this.connections.self.gain *
          this.connections.self.weight *
          connection.xtrace.values[index] + this.derivative *
          connection.elegibility *
          influence
      else:
        connection.xtrace.nodes.insert(this)
        connection.xtrace.values.insert(this.derivative * connection.elegibility * influence)
  
  return this.activation

proc activate*(this: Node; input: float): float =
  this.activation = input
  return input

proc activate*(this: NodeGroup): seq[float] =
  result = newSeq[float]()
  
  for node in this.nodes:
    result.insert(node.activate())

proc activate*(this: NodeGroup; values: seq[float]): seq[float] =
  assert(values.len == this.nodes.len)

  result = newSeq[float]()
  
  for i in 0..<this.nodes.len:
    let node = this.nodes[i]
    result.insert(node.activate(values[i]))

proc noTraceActivate*(this: Node): float =
  # All activation sources coming from the this itself
  this.state = this.connections.self.gain * this.connections.self.weight * this.state + this.bias

  # Activation sources coming from connections
  for connection in this.connections.inbound:
    this.state += connection.nodeFrom.activation * connection.weight * connection.gain

  # Squash the values received
  this.activation = this.squash.forward(this.state)

  for connection in this.connections.gated:
    connection.gain = this.activation
  
  return this.activation

proc noTraceActivate*(this: Node; input: float): float = this.activate(input)

proc propagate*(this: Node; rateOption, momentumOption: Option[float]; update: bool; target: float) =
  let
    momentum = if momentumOption.isSome: momentumOption.get else: 0.0
    rate = if rateOption.isSome: rateOption.get else: 0.3
  
  var error = 0.0

  if this.kind == NodeKind.Output: # Output nodes get their error from the enviroment
    this.error.responsibility = target - this.activation
    this.error.projected = this.error.responsibility
  else:
    for connection in this.connections.outbound:
      let this = connection.nodeTo
      error += this.error.responsibility * connection.weight * connection.gain
  
    # Projected error responsibility
    this.error.projected = this.derivative * error

    # Error responsibilities from all connections gated by this neuron
    error = 0.0

    for connection in this.connections.gated:
      let this = connection.nodeTo
      
      var influence = if this.connections.self.gater == this: this.old else: 0
      
      influence += connection.weight * connection.nodeFrom.activation
      error += this.error.responsibility * influence

    this.error.gated = this.derivative * error

    this.error.responsibility = this.error.projected + this.error.gated
  
  if this.kind == NodeKind.Constant: return

  for connection in this.connections.inbound:
    var gradient = this.error.projected * connection.elegibility

    for i in 0..<connection.xtrace.nodes.len:
      let
        this = connection.xtrace.nodes[i]
        value = connection.xtrace.values[i]
      
      gradient += this.error.responsibility * value

    let deltaWeight = rate * gradient * this.mask
    connection.totalDeltaWeight = deltaWeight
    if update:
      connection.totalDeltaWeight += momentum * connection.previousDeltaWeight
      connection.weight += connection.totalDeltaWeight
      connection.previousDeltaWeight = connection.totalDeltaWeight
      connection.totalDeltaWeight = 0
  
  let deltaBias = rate * this.error.responsibility
  this.totalDeltaBias += deltaBias
  if update:
    this.totalDeltaBias += momentum * this.previousDeltaBias
    this.bias += this.totalDeltaBias
    this.previousDeltaBias = this.totalDeltaBias
    this.totalDeltaBias = 0

proc propagate*(this: NodeGroup; rateOption, momentumOption: Option[float]; target: float): seq[float] =  
  for node in this.nodes:
    node.propagate(rateOption, momentumOption, true, target)

proc isProjectingTo(this, target: Node): bool =
  if target == this and this.connections.self.weight != 0:
    return true
  
  for connection in this.connections.outbound:
    if connection.nodeTo == target:
      return true
  
  return false

proc connect*(this, target: Node; weight: float = 1): seq[Connection] =
  result = newSeq[Connection]()

  if this == target:
    if this.connections.self.weight != 0:
      echo "This connection already exists!"
    else:
      this.connections.self.weight = weight

    result.insert(this.connections.self)
  elif this.isProjectingTo(target):
    raiseAssert("Already projecting a connection to this node!")
  else:
    let connection = newConnection(this, target, weight)
    target.connections.inbound.insert(connection)
    this.connections.outbound.insert(connection)
    
    result.insert(connection)

proc connect*(this: Node; target: NodeGroup; weight: float = 1): seq[Connection] =
  result = newSeq[Connection]()

  for subNode in target.nodes:
    let connection = newConnection(this, subNode, weight)
    subNode.connections.inbound.insert(connection)
    this.connections.outbound.insert(connection)
    target.connections.inbound.insert(connection)

    result.insert(connection)

proc ungate*(this: Node; connection: Connection) =
  let index = this.connections.gated.find(connection)
  this.connections.gated.delete(index)
  connection.gater = nil
  connection.gain = 1

proc ungate*(this: Node; connections: seq[Connection]) =
  for connection in connections:
    this.ungate(connection)

proc disconnect*(this, node: Node; twosided: bool = false) =
  if this == node:
    this.connections.self.weight = 0
    return
  
  for i in countdown(this.connections.outbound.len - 1, 0):
    let connection = this.connections.outbound[i]
    if connection.nodeTo == node:
      this.connections.outbound.delete(i)
      let j = connection.nodeTo.connections.inbound.find(connection)
      connection.nodeTo.connections.inbound.delete(j)
      if connection.gater != nil:
        connection.gater.ungate(connection)
      break
  
  if twosided:
    node.disconnect(this)

proc gate*(this: Node; connection: Connection) =
  this.connections.gated.insert(connection)
  connection.gater = this

proc gate*(this: Node; connections: seq[Connection]) =
  for connection in connections:
    this.gate(connection)

proc clear*(this: Node) =
  for connection in this.connections.inbound:
    connection.elegibility = 0
    connection.xtrace.nodes = @[]
    connection.xtrace.values = @[]

  for connection in this.connections.gated:
    connection.gain = 0
  
  this.error.responsibility = 0
  this.error.projected = 0
  this.error.gated = 0

  this.old = 0
  this.state = 0
  this.activation = 0

proc mutate*(this: Node; kind: MutationKind) =
  case kind
  of MutationKind.Activation:
    let 
      current = ActivationMutation.allowed.find(this.squash)
      newSquash = (current.float + floor(rand(1.0) * (ActivationMutation.allowed.len - 1).float) + 1).int mod ActivationMutation.allowed.len
    this.squash = ActivationMutation.allowed[newSquash]
  of MutationKind.Bias:
    this.bias = rand(1.0) * (BiasMutation.max - BiasMutation.min) + BiasMutation.min
  else:
    discard

proc isProjectedBy*(this, node: Node): bool =
  if node == this and this.connections.self.weight != 0:
    return true
  
  for connection in this.connections.inbound:
    if connection.nodeFrom == node:
      return true
  
  return false