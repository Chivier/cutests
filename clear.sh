#!/usr/bin/bash
for file in */ ; do
  if [[ -d "$file" && ! -L "$file" ]]; then
    rm -rf $file/build
  fi;
done
