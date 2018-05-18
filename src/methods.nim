import math, hashes

type
  MutationKind* {.pure.} = enum
    Activation, Bias

  BiasMutation* = object
  ActivationMutation* = object

  Activation* = object
    forward*: proc(x: float): float
    derivative*: proc(x: float): float

func logisticF(x: float): float = 1 / (1 + exp(-x))
func logisticD(x: float): float =
  let fx = logisticF(x)
  fx * (1 - fx)

func tanhF(x: float): float = tanh(x)
func tanhD(x: float): float = 1 - pow(tanh(x), 2)

func identityF(x: float): float = x
func identityD(x: float): float = 1

func reluF(x: float): float = return if x > 0.0: x else: 0.0
func reluD(x: float): float = return if x > 0.0: 1.0 else: 0.0

template logistic*(_: typedesc[Activation]): Activation = Activation(forward: logisticF, derivative: logisticD)
template tanh*(_: typedesc[Activation]): Activation = Activation(forward: tanhF, derivative: tanhD)
template identity*(_: typedesc[Activation]): Activation = Activation(forward: identityF, derivative: identityD)
template relu*(_: typedesc[Activation]): Activation = Activation(forward: reluF, derivative: reluD)

var
  mutationBiasMin = -1.0
  mutationBiasMax = 1.0
  
  allowedActivationMutations = @[
    Activation.logistic,
    Activation.tanh,
    Activation.identity,
    Activation.relu
  ]

proc min*(_: typedesc[BiasMutation]): float = mutationBiasMin
proc min*(_: typedesc[BiasMutation], value: float) = mutationBiasMin = value
proc max*(_: typedesc[BiasMutation]): float = mutationBiasMax
proc max*(_: typedesc[BiasMutation], value: float) = mutationBiasMax = value

proc allowed*(_: typedesc[ActivationMutation]): seq[Activation] = allowedActivationMutations
