# 21F1006125_IITMBS_MLOPS_OPPE2_MAY_2026_MOCK

## setup python
* Run 'python3 -m venv .env'
* Run 'source .env/bin/activate'
* Run 'pip install -r requirements.txt'

## Setup Kubernetes
* Create a cluster with default settings -> Takes about 5 min
* Create a workload from the existing image
* Expose it via load balancer

## Steps to start MLFlow instance
* Open the workbench instance in SSH mode
* Install mlflow library (```pip install mlflow```)
* Create a new screen (screen -S mlflow_experiment)
* Start mlflow server
  ```
  mlflow server \
    --host 0.0.0.0 \
    --port 8100 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*"
  ```
* Press keys Ctrl + A and Ctrl + D to detach from screen
* To list the existing screens, use 'screen -list'
* To reattach to previous screen, use 'screen -R mlflow_experiment)
* Create a firewall rule to allow mlflow instance (External IP address of VPC instance, port: 8100)
* Get the external IP address of the VM instance and access the IP (Say 35.202.51.100:8100) -> MLFlow UI page displays

## Stress testing commands
* Install wrk library
```sudo apt-get install -y wrk```
* Create a lua file (Say with file name - stress-test.lua)
```
wrk.method = "POST"

wrk.body = [[
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
]]

wrk.headers["Content-Type"] = "application/json"
```
* Run the following command
```wrk -t4 -c1000 -d30s  -s stress-test.lua http://34.172.240.72:80/predict```