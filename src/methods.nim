import math

type
  MutationKind* {.pure.} = enum
    Activation, Bias

  ConnectionKind* {.pure.} = enum
    AllToAll, AllToElse, OneToOne

  GatingKind* {.pure.} = enum
    Output, Input, Self

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
  BiasMutation* = (min: -1.0, max: 1.0)
  ActivationMutation* = (
    allowed: @[
      Activation.logistic,
      Activation.tanh,
      Activation.identity,
      Activation.relu
    ])
