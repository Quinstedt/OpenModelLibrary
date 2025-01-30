## Get started
### Load module
Start by loading the module (see [modules documentation](https://www.c3se.chalmers.se/documentation/modules/))

**Example:**
```ml load Python/3.11.5-GCCcore-13.2.0```
## Create environment using apptainer
Create an Apptainer ( see [container documentation](https://www.c3se.chalmers.se/documentation/applications/containers/))

**Github:** [C3SE container examples](https://github.com/c3se/containers/)

Two files are required: 
1. `my_recipe.def` - The Definition File: is a text file that contains the instructions for building the container. It specified: 
- The base image to use (e.g., Ubuntu, CentOS)
- Software packages to install
- Environment configurations and setup commands

**Example: ***
```
Bootstrap: localimage
From: /apps/containers/Conda/miniconda-latest.sif
#From: /apps/containers/Conda/miniconda-23.10.0-1.sif

%files 
    /cephyr/NOBACKUP/groups/llm-readability/setup/requirements.text

%post
    /opt/conda/bin/conda install pip
    /opt/conda/bin/pip install -r /cephyr/NOBACKUP/groups/llm-readability/setup/requirements.text
```


The example above uses the miniconda.sif as a base image and copies the _requirements.text_ file from the host system into the container and then install the Python packages listed in the _requirements.text_ file into the container.


2. `my_container.sif` - The Image File: is the actual container image you run. It contains the full environment, including the operating system, installed software, and dependencies.

### How to build and execute
```apptainer build my_container_name.sif my_recipe.def```

The my_container_name can be changed to any name we want. 

To use the container:
`apptainer exec <whole path>/mycontainer.sif`

**Example:** 
apptainer exec /cephyr/NOBACKUP/groups/myProjectName/MyFolder/mycontainer.sif 

### Simplify usage

The paths in C3SE are long and hard to memorize therefore using alias can help. 
In _HOME Directory_ create .bashrc file to handle shortcuts example:

#### create a .bashrc

```
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# Bash aliases

alias ls="ls --color=auto"
alias ll="ls -alhF"

# Using Python Environment
# Load Module and activate Python environment - not recommended due the file number constrain in C3SE
alias activate="ml load Python/3.11.5-GCCcore-13.2.0 && source /cephyr/NOBACKUP/groups/MyProject/.venvs/myPythonEnv/bin/activate"

# Shortcut to the project
alias myproject="cd /cephyr/NOBACKUP/groups/myProjectName"

# Shortcut to the project folder
alias mySubFolder="cd /cephyr/NOBACKUP/groups/myProjectName/MyFolder"

# Using Apptainer 
# To execute anything in the apptainer environment
alias myenv="apptainer exec /cephyr/NOBACKUP/groups/myProjectName/MyFolder/mycontainer.sif"

# Nested Alias
# Shortcut to execute a command with the apptainer using a precious alias
alias myenvpy="mistral python"
```

Now instead of using: 
`apptainer exec /cephyr/NOBACKUP/groups/myProjectName/MyFolder/mycontainer.sif` we can use `mistral`

#### create.bash_profile

```
#.bash_profile
#Get the aliases and functions
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
# User specific environment and startup programs
```
This script checks if the ~/.bashrc file exists and, if it does, sources it to ensure that aliases, functions, and configurations from .bashrc are applied in login shells as well



