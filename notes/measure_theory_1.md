markdown = r"""# Concept Map: Sigma-Algebras and Random Variables

## 1. The big picture

Probability theory starts with a mathematical model of uncertainty. The model has three basic ingredients:

- **A sample space** \(\Omega\): the set of all possible outcomes.
- **A sigma-algebra** \(\mathcal{F}\): the collection of events to which we assign probabilities.
- **A probability measure** \(P\): a function that assigns a probability to each event in \(\mathcal{F}\).

A **random variable** is then a measurable function that turns an outcome in \(\Omega\) into a numerical value (or, more generally, a value in another measurable space).

The key connection is:

> A sigma-algebra specifies which questions about an outcome count as observable events. A random variable is measurable when asking whether its value lies in a measurable set corresponds to an event in that sigma-algebra.

## 2. The foundational objects

### 2.1 Sample space: \(\Omega\)

The sample space contains every possible outcome of an experiment.

Examples:

- One coin toss: \(\Omega = \{H,T\}\).
- Two die rolls: \(\Omega = \{1,\ldots,6\}^2\).
- A measurement of waiting time: \(\Omega = [0,\infty)\), in a suitable model.

An individual outcome is often written \(\omega\in\Omega\).

### 2.2 Events and sigma-algebras: \(\mathcal{F}\)

An **event** is a subset of \(\Omega\). For example, when rolling a die, the event “the result is even” is \(\{2,4,6\}\).

A collection \(\mathcal{F}\subseteq 2^\Omega\) is a **sigma-algebra** if it satisfies:

1. \(\Omega\in\mathcal{F}\).
2. If \(A\in\mathcal{F}\), then its complement \(A^c=\Omega\setminus A\) is in \(\mathcal{F}\).
3. If \(A_1,A_2,\ldots\in\mathcal{F}\), then \(\bigcup_{n=1}^{\infty}A_n\in\mathcal{F}\).

These properties also imply that \(\varnothing\in\mathcal{F}\) and that sigma-algebras are closed under countable intersections.

The elements of \(\mathcal{F}\) are the **measurable events**. The sigma-algebra is not necessarily the collection of every subset of \(\Omega\); it records which subsets the model treats as events.

### 2.3 Probability measure: \(P\)

A probability measure is a function

\[
P:\mathcal{F}\longrightarrow[0,1]
\]

such that:

- \(P(\Omega)=1\).
- \(P(A)\geq 0\) for every \(A\in\mathcal{F}\).
- For pairwise disjoint events \(A_1,A_2,\ldots\),
  \[
  P\left(\bigcup_{n=1}^{\infty}A_n\right)
  =\sum_{n=1}^{\infty}P(A_n).
  \]

The triple \((\Omega,\mathcal{F},P)\) is called a **probability space**.

Notice the order of construction: first specify the outcomes, then the measurable events, and then the probabilities of those events.

## 3. Why do we need sigma-algebras?

It may seem natural to assign probabilities to every subset of \(\Omega\). In finite probability spaces this is usually possible. In more general spaces, especially uncountable ones such as \(\mathbb{R}\), assigning probabilities consistently to every subset can be impossible or mathematically inconvenient.

A sigma-algebra gives a carefully chosen collection of sets on which probability is defined. It is also the language used to express what information is available.

For example, if \(\Omega\) represents all possible outcomes of an experiment, a sigma-algebra \(\mathcal{F}\) describes the events that can be distinguished or observed within the model.

## 4. Random variables are measurable functions

Let \((\Omega,\mathcal{F})\) be a measurable space and let \((S,\mathcal{S})\) be another measurable space. A function

\[
X:\Omega\longrightarrow S
\]

is a **random variable** (more generally, a measurable map) if, for every \(B\in\mathcal{S}\),

\[
X^{-1}(B)
=\{\omega\in\Omega:X(\omega)\in B\}
\in\mathcal{F}.
\]

Here:

- \(S\) is the value space.
- \(\mathcal{S}\) is the sigma-algebra of measurable subsets of \(S\).
- \(X^{-1}(B)\) is the **preimage** of \(B\): the set of outcomes whose values under \(X\) fall in \(B\).

For a real-valued random variable, the usual value space is \((\mathbb{R},\mathcal{B}(\mathbb{R}))\), where \(\mathcal{B}(\mathbb{R})\) is the Borel sigma-algebra.

Thus, a real-valued function is a random variable when

\[
\{\omega:X(\omega)\in B\}\in\mathcal{F}
\quad\text{for every }B\in\mathcal{B}(\mathbb{R}).
\]

An equivalent test for real-valued \(X\) is that

\[
\{\omega:X(\omega)\leq x\}\in\mathcal{F}
\quad\text{for every }x\in\mathbb{R}.
\]

### Intuition

A random variable does not mean that the function itself is random in the ordinary sense. It is a function defined on the outcome space. The uncertainty comes from the random outcome \(\omega\). Measurability guarantees that statements about the value of \(X\), such as “\(X\leq 3\),” correspond to events that belong to \(\mathcal{F}\).

## 5. A concrete example: rolling a fair die

Let

\[
\Omega=\{1,2,3,4,5,6\},\qquad
\mathcal{F}=2^\Omega.
\]

Since the sample space is finite, take every subset to be measurable. Let each outcome have probability \(1/6\).

Define the random variable

\[
X(\omega)=
\begin{cases}
1,&\omega\in\{1,3,5\},\\
0,&\omega\in\{2,4,6\}.
\end{cases}
\]

This variable records whether the die result is odd (1) or even (0).

For example,

\[
X^{-1}(\{1\})=\{1,3,5\}\in\mathcal{F}.
\]

Therefore the event “\(X=1\)” is measurable, and

\[
P(X=1)=P(\{1,3,5\})=\frac{3}{6}=\frac12.
\]

Similarly, \(P(X=0)=1/2\). The random variable translates outcomes of the die experiment into values in \(\{0,1\}\).

## 6. The distribution of a random variable

A measurable random variable allows us to transfer the probability measure from the original outcome space to its value space.

For \(X:\Omega\to S\), define its **distribution** (or pushforward probability measure) by

\[
P_X(B)=P(X^{-1}(B)),\qquad B\in\mathcal{S}.
\]

In words, the probability that \(X\) takes a value in \(B\) is the probability of all original outcomes that are mapped into \(B\).

For a real-valued random variable, its cumulative distribution function (CDF) is

\[
F_X(x)=P(X\leq x)
=P(\{\omega:X(\omega)\leq x\}).
\]

The distribution describes the random variable's probabilities without needing to refer to the original sample space in every calculation.

## 7. Sigma-algebras as information

A sigma-algebra can also be interpreted as a body of information.

Suppose \(\mathcal{G}\subseteq\mathcal{F}\) is a sigma-algebra. Then \(\mathcal{G}\) contains no more information than \(\mathcal{F}\): it may distinguish fewer events.

A random variable \(X\) is **\(\mathcal{G}\)-measurable** if

\[
\{\omega:X(\omega)\in B\}\in\mathcal{G}
\quad\text{for every measurable }B.
\]

This means that the value of \(X\) can be determined using the information represented by \(\mathcal{G}\).

For example, imagine that the full outcome records both dice in a pair of rolls, but the available information reveals only their sum. The sigma-algebra generated by the sum captures exactly the events that can be determined from that sum.

### The sigma-algebra generated by a random variable

The notation

\[
\sigma(X)
\]

means the smallest sigma-algebra on \(\Omega\) that makes \(X\) measurable. For real-valued \(X\),

\[
\sigma(X)=
\{X^{-1}(B):B\in\mathcal{B}(\mathbb{R})\},
\]

with the understanding that this collection is a sigma-algebra.

It represents the information revealed by observing \(X\). If \(\sigma(X)\subseteq\mathcal{G}\), then \(X\) is \(\mathcal{G}\)-measurable.

## 8. The concept map

The relationships can be summarized as follows:

```text
                         PROBABILITY MODEL
                                |
                                v
                    Sample space: Ω
                    (possible outcomes)
                                |
                                v
                 Sigma-algebra: F on Ω
                 (measurable events / information)
                                |
                                v
                 Probability measure: P
                 (probabilities of events)
                                |
                                v
                    Probability space
                       (Ω, F, P)
                                |
                                | measurable function
                                v
                  Random variable: X
                     X : Ω → S
                                |
                measurability condition:
              X⁻¹(B) belongs to F for all B
                                |
                                v
                 Distribution of X: P_X
              P_X(B) = P(X⁻¹(B))
                                |
                                v
              Probabilities about the values of X
```

A second way to view the information relationship is:

```text
       Full information: F
              |
              | contains at least as much information as
              v
       Information in X: σ(X)
              |
              | generated by observing
              v
       Random variable: X
```

## 9. Common points of confusion

- **A random variable is a function, not a value.** The value \(X(\omega)\) is obtained after an outcome \(\omega\) is specified.
- **An event is a set of outcomes.** For example, \(\{X\leq x\}\) is shorthand for \(\{\omega:X(\omega)\leq x\}\).
- **Measurability is a condition on preimages.** It asks whether inverse images of measurable value sets are measurable events in the original space.
- **The sample space and the value space are different.** The sample space describes the experiment; the value space describes what the random variable records.
- **The distribution is not the random variable itself.** The function \(X\) maps outcomes to values, while \(P_X\) assigns probabilities to sets of values.
- **A finer sigma-algebra contains more information.** If \(\mathcal{G}\subseteq\mathcal{F}\), then \(\mathcal{F}\) can distinguish at least all the events that \(\mathcal{G}\) can.

## 10. A compact checklist

When reading a probability problem, ask:

1. What is the sample space \(\Omega\)?
2. Which events are measurable, i.e. what is \(\mathcal{F}\)?
3. What probability measure \(P\) is being used?
4. What function \(X\) maps outcomes to values?
5. What is the value space and its sigma-algebra?
6. Are the relevant preimages under \(X\) in \(\mathcal{F}\)?
7. What distribution \(P_X\) does \(X\) induce?

If you can answer these questions, you can usually identify how the sigma-algebra, random variable, and probability measure fit together.
