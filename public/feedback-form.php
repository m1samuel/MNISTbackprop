<?php
$name=$_POST['name'];
$email=$_POST['email'];
$comment=$_POST['comment'];

$msg ="Name: $name Email: $email Comment: $comment\n\n"; 
$recipient = "email@address"; 
$subject = "Website Feedback"; 
mail($recipient, $subject, $msg, "From: email@address","-f email@address"); 
PRINT "Thank You - Visit Us Again!!";
?>


