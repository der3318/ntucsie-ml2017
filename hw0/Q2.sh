javac -d bin -sourcepath src -encoding utf8 src/launch/Main.java
java -Dfile.encoding=UTF-8 -cp bin launch.Main 2 $1
