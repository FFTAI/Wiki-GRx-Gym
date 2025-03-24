# create related sh files
echo $CONDA_PREFIX
echo $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

# change the content of the files
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CONDA_PREFIX/lib" \
    > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

cat <<EOL > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
ORIGINAL_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
DIRECTORY_TO_REMOVE=$CONDA_PREFIX/lib
NEW_LD_LIBRARY_PATH=\$(echo \$LD_LIBRARY_PATH | tr ':' '\n' | grep -v "\$DIRECTORY_TO_REMOVE" | tr '\n' ':' | sed 's/:$//')
export LD_LIBRARY_PATH=\$NEW_LD_LIBRARY_PATH
EOL
