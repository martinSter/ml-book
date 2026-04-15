# *****************************************************
# ------------- ANN EINFACHE REGRESSION ---------------
#
# Fachhochschule Nordwestschweiz
# Riggenbachstrasse 16
# 4600 Olten
#
# Autor: Martin Sterchi
# Beschreibung: BASE R Implementierung eines einfachen ANNs
#
# *****************************************************
# 1. Hyperparameter definieren ------------------------

# Anzahl Epochen
epochs <- 100000

# Lernrate GD (alpha)
learning_rate <- 0.01


# *****************************************************
# 2. Daten generieren ---------------------------------

# 100 zufällige Trainingsdatenpunkte generieren
set.seed(42)
n <- 100
x <- runif(n, 0, 9)
y <- sin(x) + 1.5 + rnorm(n, 0, 0.1)

# Input-Daten Matrix
X <- rbind(1, x)


# *****************************************************
# 3. Funktionen definieren ----------------------------

# Sigmoid Aktivierung
sigmoid <- function(z) {1 / (1 + exp(-z))}

# Forward-Pass
forward <- function(X, W1, w2) {
  Z  <- W1 %*% X
  A  <- sigmoid(Z)
  A1 <- rbind(1, A)
  t  <- w2 %*% A1
  list(Aktivierung = A, Aktivierung1 = A1, Output = t)
}

# Gradienten erster und zweiter Layer
gradient_first_layer <- function(n, w2, y, t, A, X) {
  -1/n * ((w2[2:length(w2)] %*% (y - t) * A * (1 - A)) %*% t(X))
}

gradient_second_layer <- function(n, y, t, A1) {
  -1/n * (y - t) %*% t(A1)
}


# *****************************************************
# 4. Backpropagation ----------------------------------

# Gewichte zufällig initialisieren
W1 <- matrix(rnorm(6), nrow = 3)
w2 <- rnorm(4)

# Leeren Vektor für Kostenwerte (Loss) initialisieren
loss <- vector("double", epochs)

# Iteration über Epochen
for (i in 1:epochs) {
  # Forward-Pass
  out <- forward(X, W1, w2)
  # Aktuelle Kosten
  loss[i] <- mean((out$Output - y)^2)
  # Backward-Pass
  g1 <- gradient_first_layer(n, w2, y, out$Output, out$Aktivierung, X)
  g2 <- gradient_second_layer(n, y, out$Output, out$Aktivierung1)
  # Gradient Descent Schritte
  W1 <- W1 - learning_rate * g1
  w2 <- w2 - learning_rate * g2
}


# *****************************************************
# 5. Visualisierung -----------------------------------

# Plot Layout
layout(mat = matrix(c(1, 1, 2, 3), nrow = 2, ncol = 2, byrow = TRUE),
       widths  = c(2, 2),
       heights = c(6, 4))

# Sequenz für Plots
x_seq  <- seq(0, 9, 0.01)
X_plot <- rbind(1, x_seq)

# ---- Plot 1: Daten & Modell ----
par(mar = c(5, 4, 0.5, 0.5))
plot(0, 0, type = "n", xlab = "x", ylab = "y",
     xlim = c(0, 9), ylim = c(0, 3),
     cex.axis = 1.3, cex.lab = 1.3)
grid()
points(x, y, col = "steelblue", pch = 20, cex = 1.5)
lines(x_seq, forward(X_plot, W1, w2)$Output, col = "rosybrown3", lwd = 3)

# ---- Plot 2: Kosten pro Epoche ----
plot(seq(1, epochs, 1), loss, type = "l", col = "rosybrown3", lwd = 3,
     xlab = "Epochen", ylab = "Kosten",
     log = "x", cex.axis = 1.3, cex.lab = 1.3)
grid()

# ---- Plot 3: Aktivierungen ----
act <- sigmoid(W1 %*% X_plot)
plot(0, 0, type = "n", yaxt = "n", xlab = "x", ylab = "Aktivierungen",
     xlim = c(0, 9), ylim = c(0, 1), cex.axis = 1.3, cex.lab = 1.3)
grid()
lines(x_seq, act[1, ], col = "rosybrown3", lwd = 3, lty = 1)
lines(x_seq, act[2, ], col = "rosybrown3",       lwd = 3, lty = 2)
lines(x_seq, act[3, ], col = "rosybrown3",       lwd = 3, lty = 3)
axis(2, at = seq(0, 1, 0.5), lwd = 0, lwd.ticks = 1, cex.axis = 1.3)
legend(x = "topright",
       legend = c(expression(g(z[1]^{(1)})),
                  expression(g(z[2]^{(1)})),
                  expression(g(z[3]^{(1)}))),
       lty = c(1, 2, 3),
       col = c("rosybrown3", "rosybrown3", "rosybrown3"),
       lwd = 3, cex = 1.1)

