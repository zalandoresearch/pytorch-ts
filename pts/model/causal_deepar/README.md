# Causal `DeepAR`

Causal `DeepAR` model augments the `DeepAR` model by incorporating a causal structure via a `control` time-dependent signal.

The main assumption of this model is that the `target` at time `t` depends not only   on the covariates up till time `t` and but also on `control` till time `t`. Thus we encode this structure via a model which now adds a `control_output` distribution layer. The main assumption we have is that at training time our dataset of time series now have an additional `control` key with the corresponding 1-d control-variate values (with the array being as large as `target`).

We translate this structure into the following schematics of Causal-`DeepAR` at time `t`:

```
┌─────────┐             ┌────────┐ 
│control_t│──────┐      │target_t│ 
└─────────┘      │      └────────┘ 
     ▲           │           ▲     
     │           │           │     
 log-prob/       │       log-prob/ 
   sample        │         sample  
     │      ground-truth/    │     
┌────────┐     sample/   ┌───────┐ 
│control │       do      │target │ 
│  dist  │       └──────▶│ dist  │ 
└────────┘               └───────┘ 
     ▲                       ▲     
     └────────────┬──────────┘     
                  │                
               ┌────┐              
     ───h_t-1─▶│RNN │───h_t──▶     
               └────┘              
                  ▲                
            ┌─────┴────┐           
            │target_t-1│           
            └─────┬────┘           
            ┌─────┴─────┐          
            │control_t-1│          
            └─────┬─────┘          
              ┌───┴──┐            
              │cov_t │            
              └──────┘  
```


The model is trained as per the DeepAR assumption, which mainly implies that the covariates and `control` are known for all the time points while training and only the covariates are known for all the time points that we wish to forecast for.

In terms of the `control` at inference time, we now have two choices:

1. We can predict by setting the `control` array to `np.Nan` for the duration of the prediction length in which case the model will sample values from the `control_output.distribution` and feed it auto-regressively to the `target` head and RNN.

2. On the other hand we can in the prediction window "do" an intervention by setting the values of `control` to some fixed value of our choosing for the time steps in the future we are interested in. In this case this model will "break" the causal connections and just use the supplied values and feed those to the `target` head and RNN at the appropriate time points.

## Possible uses

* `target` can be sales and `control` can be discounts
* `target` can be sales and `control` can be the temperature
  
