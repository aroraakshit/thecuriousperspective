Source for [aroraakshit.github.io/thecuriousperspective](https://aroraakshit.github.io/thecuriousperspective/), based on [jekyllt/jasper2](https://github.com/jekyllt/jasper2).

Pre-requisites (especially for Mac users with Apple Silicon):

This project relies on Jekyll 3.9, which requires an older version of Ruby (2.7.x) than what comes installed on modern systems.

1. **Install a Ruby version manager and dependencies**
   ```bash
   brew install rbenv ruby-build libffi libyaml
   ```

2. **Initialize rbenv in your shell** (Run once and restart terminal)
   ```bash
   echo 'eval "$(rbenv init -)"' >> ~/.zshrc
   source ~/.zshrc
   ```

3. **Install and set local Ruby version**
   ```bash
   rbenv install 2.7.6
   rbenv local 2.7.6
   ```

4. **Install Bundler and project dependencies** 
   *(Note: You may need custom flags for `ffi` on Apple Silicon Macs)*
   ```bash
   gem install bundler -v 1.17.2
   bundle config build.ffi --with-cflags="-Wno-error=implicit-function-declaration" --with-ldflags="-L$(brew --prefix libffi)/lib" --with-cppflags="-I$(brew --prefix libffi)/include"
   bundle install
   ```

5. **Start the local server**
   ```bash
   bundle exec jekyll serve --port 8000 --host localhost 
   ```

References:
* [Ghost.org](https://ghost.org/)