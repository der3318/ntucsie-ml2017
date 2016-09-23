javac -d bin -sourcepath src -encoding utf8 src/Main.java
java -Dfile.encoding=UTF-8 -cp bin Main 1 $1 $2
