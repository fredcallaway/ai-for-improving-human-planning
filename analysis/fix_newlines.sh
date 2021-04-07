for file in [1-8]/*.tex; do
    cp $file $file.bak
    perl -p -i -e 's/\R//g;' $file
    echo >> $file
done