(define (problem block_words)
	(:domain block-words)
	(:objects
		p o w e r c - block
	)
	(:init
		(handempty)

		(clear w)
		(on w p)
		(ontable p)

		(clear o)
		(on o e)
		(on e c)
		(ontable c)

<<<<<<< HEAD
		(ontable r)
		(clear r)
	)
	(:goal (and
		;; power
		(clear p) (ontable c) (on p o) (on o w) (on w e) (on e r)
=======
		(clear r)
		(ontable r)
	)
	(:goal (and
		;; power
		(clear p) (ontable r) (on p o) (on o w) (on w e) (on e r)
>>>>>>> 0e2fda6029f6d6e5aac693dc9bb7ab5b588bcc0d
	))
)
