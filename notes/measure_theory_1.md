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

