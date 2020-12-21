import Cairo, Fontconfig
using Yao, YaoPlots, Compose

plot(kron(X, X)) |> PNG("kron-X-X.png")

plot(chain(X, X)) |> PNG("chain-X-X.png")
