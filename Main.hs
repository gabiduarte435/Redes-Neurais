module Main where

import Control.Monad (forM_)

-- Derivadas
grad :: Double -> Double -> (Double, Double)
grad w b =
  let dedw = 2 * ((5 - w - b) * (-1) + (7 - 2 * w - b) * (-2) + (9 - 3 * w - b) * (-3) + (10 - 4 * w - b) * (-4) + (12 - 5 * w - b) * (-5) + (15 - 6 * w - b) * (-6))
      dedb = -2 * ((5 - w - b) + (7 - 2 * w - b) + (9 - 3 * w - b) + (10 - 4 * w - b) + (12 - 5 * w - b) + (15 - 6 * w - b))
   in (dedw, dedb)

-- Função do Erro
descent :: Double -> Double -> Double -> Double -> Int -> (Double, Double, Int)
descent lr xt yt err i
  | err < tol = (xt, yt, i)
  | otherwise =
      let dfdx = fst (grad xt yt)
          dfdy = snd (grad xt yt)
          xnovo = xt - lr * dfdx
          ynovo = yt - lr * dfdy
          errnovo = sqrt ((xnovo - xt) ** 2 + (ynovo - yt) ** 2)
       in descent lr xnovo ynovo errnovo (i + 1)

predict :: Double -> [Double] -> Double
predict xi xs =
  let w1 = xs !! 0
      w2 = xs !! 1
      b1 = xs !! 2
      b2 = xs !! 3
   in sigma (w2 * sigma (w1 * xi + b1) + b2)

-- Função sigmoide
sigma :: Double -> Double
sigma x = 1 / (1 + exp (-x))

func :: Int -> Double
func n
  | n >= 500 = 0.5
  | otherwise = 0.5

neural :: [(Double, Double)] -> [Double] -> [Double]
neural ts xs =
  let w1 = xs !! 0
      w2 = xs !! 1
      b1 = xs !! 2
      b2 = xs !! 3
      f xi = sigma (w2 * sigma (w1 * xi + b1) + b2)
      dedw1 =
        sum
          [ -(yi - f xi)
              * f xi
              * (1 - f xi)
              * w2
              * sigma (w1 * xi + b1)
              * (1 - sigma (w1 * xi + b1))
              * xi
            | (xi, yi) <- ts
          ]
      dedw2 =
        sum
          [ -(yi - f xi)
              * f xi
              * (1 - f xi)
              * sigma (w1 * xi + b1)
            | (xi, yi) <- ts
          ]
      dedb1 =
        sum
          [ -(yi - f xi)
              * f xi
              * (1 - f xi)
              * w2
              * sigma (w1 * xi + b1)
              * (1 - sigma (w1 * xi + b1))
            | (xi, yi) <- ts
          ]
      dedb2 =
        sum
          [ -(yi - f xi)
              * f xi
              * (1 - f xi)
            | (xi, yi) <- ts
          ]
   in [dedw1, dedw2, dedb1, dedb2]

descentV :: ([Double] -> [Double]) -> Double -> Int -> Double -> [Double] -> ([Double], Int)
descentV grad lr i err xts
  | err < tol = (xts, i)
  | otherwise =
      let dfdxs = grad xts
          xsnovo = [xt - lr * grad | (xt, grad) <- zip xts dfdxs]
          errnovo = sum [(xnovo - xt) ** 2 | (xnovo, xt) <- zip xsnovo xts]
       in descentV grad lr (i + 1) errnovo xsnovo

-- Tolerância
tol :: Double
tol = 10 ** (-6)

parOuImpar :: Int -> String
parOuImpar n =
  -- mod/2 == 0 logo 0 == 1, logo impar
  if n `mod` 2 == 0 then "par" else "impar"

main = do
  putStrLn "Digite um valor: "
  input <- getLine
  let value = read input :: Double
      tc = [(fromIntegral n / 1000, func n) | n <- [1 .. 1000]]
      -- Valores Iniciais
      p = descentV (neural tc) 0.01 0 9999 [18.438781391117114, 18.28050394706888, -9.372209396659654, -8.398746300630092]
      fp = fst p
  putStrLn "Descent"
  print p
  putStrLn $ "Predição para " ++ input
  putStrLn $ if odd (round value) then "impar" else "par"
  print $ predict (value / 1000) fp

  -- Função para imprimir os valores e suas classificações(Impar-Par)
  let numbers = [988, 299, 800, 9, 17, 1000]
  forM_ numbers $ \num -> do
    putStrLn (show num)
    if predict (fromIntegral num / 1000) fp >= 0.5
      then putStrLn ("O número " ++ show num ++ " é par")
      else putStrLn ("O número " ++ show num ++ " é ímpar")