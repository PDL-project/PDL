(define (problem place-bread-in-cabinet)
  (:domain allactionrobot)

  (:objects
    robot1 - robot
    bread - object
    cabinet - object
    diningtable - object
    floor - object
  )

  (:init
    (= (total-cost) 0)
    (not (inaction robot1))
    (at robot1 floor)
    (at-location bread diningtable)
    (at-location cabinet floor)
    (not (holding robot1 bread))
    (object-close robot1 cabinet)
  )

  (:goal (and
    (at-location bread cabinet)
    (object-close robot1 cabinet)
  ))

  (:metric minimize (total-cost))
)