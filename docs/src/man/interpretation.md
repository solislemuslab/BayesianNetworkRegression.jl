# Interpreting the Model Results

## Result format

The `Fit!` function will return a `Results` object with the following members:
- state
- rhat

### State

result.state contains the (post-burn-in) outputs of the sampling algorithm, with focus on the following two variables
-  $\xi$: a vector describing whether each node (microbe) is influential on the response. Set to 1 if the microbe is influential and 0 if it is not. 
-  $\gamma$: a vector of coefficients describing the effect of each edge (relationship) on the response. 

### $\hat{R}$

result.rhat contains r-hat statistics ([Vehtari et al. 2020](http://www.stat.columbia.edu/~gelman/research/unpublished/1903.08008.pdf)), which are used to assess whether the sampling algorithm has converged. Values close to 1 indicate convergence (Vehtari et al. suggest using a cutoff of $\hat{R} < 1.01$ to indicate convergence). Values are provided for all $\xi$ and $\gamma$ variables.

## Interpretation

### State

To aid in interpretation of results, run the following (in julia):
```julia
using Statistics
mean(result.state.ξ[:,:],dims=1)
mean(result.state.γ[:,:],dims=1)
```
The first mean calculates the posterior probability that each node is influential. 
The second mean calculates the point-estimate for the coefficients.

We'll save the posterior probabilites in a file so we can use them later (`true_xi.csv` is available [here](https://github.com/samozm/BayesianNetworkRegression.jl/blob/main/examples/true_xi.csv)):

```julia
pp_df = DataFrame(CSV.File("true_xi.csv"))
pp_df[:,"Xi posterior"] = mean(result.state.ξ[:,:],dims=1)

CSV.write("nodes.csv",pp_df)
```

In order to generate (95%) credible intervals for the coefficients, the following can be run. `nsamp` should be the number of post-burn-in samples. This also pulls in the actual coefficient values (the true B matrix). The true B coefficient values can be found [here](https://github.com/samozm/BayesianNetworkRegression.jl/blob/main/examples/true_b.csv)

```julia
using DataFrames,CSV

nsamp = 10000
γ_sorted = sort(result.γ,dims=1)
lw = convert(Int64, round(nsamp * 0.025))
hi = convert(Int64, round(nsamp * 0.975))

ci_df = DataFrame(mean=mean(γ,dims=1)[1,:])
ci_df[:,"0.025"] = γ_sorted[lw,:,1]
ci_df[:,"0.975"] = γ_sorted[hi,:,1]

b_in = DataFrame(CSV.File("true_b.csv"))
B₀ = convert(Array{Float64,1},b_in[!,:B])

ci_df[:,"true_B"] = B₀

CSV.write("CIs.csv",ci_df)

ci_df
```

The following R code can be used to plot credible intervals. 

```R
flnm <- "CIs.csv" # change this to the name you saved your CI data above in 
edges <- read.csv(flnm)

pi <- 0.8
mu <- 1.6

nn <- length(edges$true_B)
edges$edge <- 1:nn
edges <- transform(edges,rej=ifelse(X0.025 > 0 | X0.975 < 0,TRUE,FALSE))
edges <- transform(edges,nonzero_mean=ifelse(true_B != 0.0,TRUE,FALSE))

label = "True influential edges"

plt <- edges %>% ggplot() + geom_errorbar(aes(x=factor(edge),ymin=X0.025,
                                                                ymax=X0.975,
                                                                color=rej)) +
        xlab("") + ylab("")+ ggtitle(paste0("pi=",pi,", mu=",mu))+
        scale_color_manual(values=c("#82AFBC","#0E4B87")) + 
        theme(
            plot.title = element_text(hjust=0.5, size=rel(2)),
            axis.title.x = element_text(size=rel(1.2)),
            axis.title.y = element_text(size=rel(1.9), angle=90, vjust=0.5, hjust=0.5),
            axis.text.x = element_text(colour="grey", size=rel(1.8), hjust=.5, vjust=.5, face="plain"),
            axis.ticks.y = element_blank(),
            panel.background = element_rect(fill = NA, color = "black"),
            axis.line = element_line(colour = "grey"),
            strip.text = element_text(size = rel(1.5)),
            legend.position = "none"
        ) +
        scale_y_continuous(limits=lim) +
        coord_flip() + scale_x_discrete(labels=NULL,expand=expansion(add=4)) +
        geom_point(aes(x=factor(edge),y=mean, color=rej),shape=18, size=2) +
        geom_hline(aes(yintercept=0),linetype="solid",color="#696969", size=1) +
        facet_grid(nonzero_mean ~.,scales="free_y",space="free_y", 
                   labeller = labeller(
                   nonzero_mean = c(`TRUE` = label, 
                                   `FALSE` = "True non-influential edges")
                   )
        ) 
plt
```

The following R code can be used to plot posterior probabilities of influence for the nodes:

```R
## Reading csv output files and appending
flnm <- "nodes.csv"
nodes <- read.csv(flnm)

pi <- 0.8
mu <- 1.6
k <- 22
n <- 100

## Labelling the nodes:
nn <- length(nodes$TrueXi) / (length(pi)*length(mu))
nodes$microbe <- rep(1:nn,length(pi)*length(mu))

ylabel = "PP of influential node"
axisTextY = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain")

## Plot:
plt <- ggplot(data=nodes,aes(x=microbe,y=Xi.posterior,fill=TrueXi)) + 
    geom_bar(stat="Identity") + xlab("") + ylab(ylabel) + 
    ggtitle(paste0("k=",k,", n=",n)) +
    theme(
        plot.title = element_text(hjust=0.5, size=rel(2)),
        axis.title.x = element_text(size=rel(1.2)),
        axis.title.y = element_text(size=rel(1.9), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        axis.text.y = axisTextY,
        panel.background = element_rect(fill = NA, color = "black"),
        axis.line = element_line(colour = "grey"),
        strip.text = element_text(size = rel(2))
    ) +
    scale_fill_manual(values=c("#82AFBC","#0E4B87")) + 
    guides(fill="none") +
    scale_y_continuous(position="right",breaks = c(0,0.5,1),limits=c(0,1))
plt
```